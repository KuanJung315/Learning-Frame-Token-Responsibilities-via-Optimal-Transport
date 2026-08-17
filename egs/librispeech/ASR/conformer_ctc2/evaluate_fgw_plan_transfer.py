#!/usr/bin/env python3
"""Reconstruct training-time FGW plans and measure transfer to CTC geometry.

For every checkpoint this evaluator constructs two plans on the same model
outputs:

1. a counterfactual OT plan with lambda_gw=0; and
2. the training-matched FGW plan with the checkpoint's lambda_gw.

It then compares both plans with the transcript-conditioned CTC occupancy and
with the lambda_gw=0 checkpoint on exactly matched cuts.  This separates the
direct effect of the structural cost on the transport plan from changes that
were actually transferred into the trained CTC posterior.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ASR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from asr_datamodule import LibriSpeechAsrDataModule
from evaluate_alignment_metrics import _checkpoint_metadata, _load_eval_dataloader
from evaluate_four_alignment_metrics import _build_vfta_model
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.utils import setup_logger, str2bool
from vfta_fgw.ot_fgw import vi_fgw_loss_v2
from shared_alignment_viz import (
    _compute_ctc_state_occupancy,
    compute_plan_agreement_metrics,
)


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    exp_dir: Path
    lambda_gw: float


@dataclass(frozen=True)
class PlanConfig:
    column_marginal_type: str
    alpha_smooth_mix: float
    bpe_col_floor: float
    token_prior_sigma: float
    token_prior_score_temp: float
    token_prior_floor: float
    eps: float
    iters: int
    beta_pos: float
    lambda_gw: float
    n_outer: int
    subsampling_factor: int


GEOMETRY_KEYS = (
    "diag_mean_abs_dev",
    "offdiag_mass",
    "support_mean_frames",
    "support_ratio",
    "column_entropy_normalized",
    "column_mass_cv",
)

AGREEMENT_KEYS = (
    "barycenter_mad",
    "support_iou",
    "total_variation",
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=EXP_DIR=LAMBDA_GW",
        help="Checkpoint to evaluate; repeat in increasing lambda order.",
    )
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--avg", type=int, default=10)
    parser.add_argument("--use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--label-embed-dim", type=int, default=256)
    parser.add_argument("--init-blank-prob", type=float, default=0.35)
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_bpe_500"))
    parser.add_argument(
        "--datasets", nargs="+", default=["dev-clean", "dev-other"]
    )
    parser.add_argument("--max-cuts-per-dataset", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--support-relative-threshold", type=float, default=0.1)
    parser.add_argument("--diagonal-band-width", type=float, default=0.12)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/fgw_plan_transfer_2000"),
    )
    LibriSpeechAsrDataModule.add_arguments(parser)
    return parser


def _parse_specs(raw_specs: Sequence[str]) -> List[CheckpointSpec]:
    specs: List[CheckpointSpec] = []
    names = set()
    for raw in raw_specs:
        parts = raw.split("=", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(
                "--checkpoint requires NAME=EXP_DIR=LAMBDA_GW, got " + raw
            )
        name, exp_dir, lambda_gw = parts
        if not name or name in names:
            raise ValueError(f"Empty or duplicate checkpoint name: {name!r}")
        names.add(name)
        specs.append(CheckpointSpec(name, Path(exp_dir), float(lambda_gw)))
    specs.sort(key=lambda spec: spec.lambda_gw)
    if not specs or not math.isclose(specs[0].lambda_gw, 0.0, abs_tol=1.0e-9):
        raise ValueError("A lambda_gw=0 checkpoint is required as the reference")
    return specs


def _checkpoint_path(spec: CheckpointSpec, epoch: int) -> Path:
    path = spec.exp_dir / f"epoch-{epoch}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _plan_config(
    spec: CheckpointSpec, epoch: int
) -> Tuple[PlanConfig, Dict[str, Any]]:
    metadata = _checkpoint_metadata(_checkpoint_path(spec, epoch))
    checkpoint_lambda = float(metadata.get("lambda_gw", 0.0))
    if not math.isclose(
        checkpoint_lambda, spec.lambda_gw, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(
            f"{spec.name}: requested lambda={spec.lambda_gw}, "
            f"checkpoint metadata says {checkpoint_lambda}"
        )
    column_type = str(metadata.get("col_marginal_type", "acoustic"))
    if column_type != "acoustic":
        raise ValueError(
            f"{spec.name}: this evaluator currently requires the acoustic "
            f"column marginal, got {column_type}"
        )
    gate_mode = str(metadata.get("gate_ctc_mode", "mixed"))
    train_prior_mix = float(metadata.get("train_prior_mix", 1.0))
    if gate_mode != "mixed" or not math.isclose(train_prior_mix, 1.0):
        raise ValueError(
            f"{spec.name}: exact evaluation-time alpha reconstruction assumes "
            f"gate_ctc_mode=mixed and train_prior_mix=1; got "
            f"{gate_mode}, {train_prior_mix}"
        )
    config = PlanConfig(
        column_marginal_type=column_type,
        alpha_smooth_mix=float(metadata.get("alpha_smooth_mix", 0.1)),
        bpe_col_floor=float(metadata.get("bpe_col_floor", 0.05)),
        token_prior_sigma=float(metadata.get("ot_token_prior_sigma", 0.15)),
        token_prior_score_temp=float(
            metadata.get("ot_token_prior_score_temp", 1.0)
        ),
        token_prior_floor=float(metadata.get("ot_token_prior_floor", 0.05)),
        eps=float(metadata.get("ot_eps", 0.3)),
        iters=int(metadata.get("ot_iters", 30)),
        beta_pos=float(metadata.get("ot_beta_pos", 1.0)),
        lambda_gw=checkpoint_lambda,
        n_outer=int(metadata.get("gw_n_outer", 3)),
        subsampling_factor=int(metadata.get("subsampling_factor", 4)),
    )
    return config, metadata


def _aligned_acoustic_features(
    feature: torch.Tensor,
    raw_num_frames: int,
    encoder_length: int,
    subsampling_factor: int,
) -> torch.Tensor:
    indices = (
        torch.arange(encoder_length, device=feature.device) * subsampling_factor
        + subsampling_factor // 2
    )
    indices = indices.clamp(max=max(raw_num_frames - 1, 0))
    return feature.index_select(0, indices)


def _support_mask(
    alignment: torch.Tensor, relative_threshold: float
) -> torch.Tensor:
    if alignment.numel() == 0:
        return torch.zeros_like(alignment, dtype=torch.bool)
    peaks = alignment.max(dim=0).values
    return (alignment >= peaks.mul(relative_threshold).unsqueeze(0)) & (
        peaks > 0
    ).unsqueeze(0)


def _alignment_geometry(
    alignment: torch.Tensor,
    relative_threshold: float,
    diagonal_band_width: float,
) -> Dict[str, float]:
    alignment = alignment.detach().float()
    T, U = alignment.shape
    if T == 0 or U == 0:
        return {key: 0.0 for key in GEOMETRY_KEYS}
    tiny = 1.0e-8
    mass = alignment.sum().clamp_min(tiny)
    t_pos = torch.linspace(0, 1, T, device=alignment.device)
    u_pos = torch.linspace(0, 1, U, device=alignment.device)
    distance = (t_pos.unsqueeze(1) - u_pos.unsqueeze(0)).abs()
    support = _support_mask(alignment, relative_threshold)
    support_width = support.sum(dim=0).float()
    column = alignment.sum(dim=0)
    column = column / column.sum().clamp_min(tiny)
    entropy = -(column * column.clamp_min(tiny).log()).sum()
    entropy_norm = entropy / math.log(U) if U > 1 else entropy.new_tensor(0.0)
    column_mean = column.mean().clamp_min(tiny)
    return {
        "diag_mean_abs_dev": float((alignment * distance).sum().div(mass).item()),
        "offdiag_mass": float(
            alignment[distance > diagonal_band_width].sum().div(mass).item()
        ),
        "support_mean_frames": float(support_width.mean().item()),
        "support_ratio": float(support_width.mean().div(max(T, 1)).item()),
        "column_entropy_normalized": float(entropy_norm.item()),
        "column_mass_cv": float(
            column.std(unbiased=False).div(column_mean).item()
        ),
    }


def _agreement(
    left: torch.Tensor,
    right: torch.Tensor,
    relative_threshold: float,
) -> Dict[str, float]:
    raw = compute_plan_agreement_metrics(
        left,
        right,
        support_relative_threshold=relative_threshold,
    )
    return {
        "barycenter_mad": raw["plan_ctc_barycenter_mad"],
        "support_iou": raw["plan_ctc_support_iou"],
        "total_variation": raw["plan_ctc_total_variation"],
    }


def _add_prefixed(
    row: Dict[str, Any], prefix: str, values: Mapping[str, float]
) -> None:
    row.update({f"{prefix}_{key}": value for key, value in values.items()})


def _reconstruct_plan(
    nonblank_log_probs: torch.Tensor,
    alpha: torch.Tensor,
    labels: torch.Tensor,
    acoustic_features: torch.Tensor,
    config: PlanConfig,
    lambda_gw: float,
) -> torch.Tensor:
    _, plan = vi_fgw_loss_v2(
        log_p_nonblank=nonblank_log_probs,
        alpha=alpha,
        labels=labels,
        acoustic_features=acoustic_features,
        column_marginal_type=config.column_marginal_type,
        alpha_smooth_mix=config.alpha_smooth_mix,
        bpe_col_floor=config.bpe_col_floor,
        token_prior_sigma=config.token_prior_sigma,
        token_prior_score_temp=config.token_prior_score_temp,
        token_prior_floor=config.token_prior_floor,
        eps=config.eps,
        iters=config.iters,
        beta_pos=config.beta_pos,
        lambda_gw=lambda_gw,
        n_outer=config.n_outer,
        return_plan=True,
    )
    if plan is None:
        raise RuntimeError("FGW plan reconstruction returned None")
    return plan


def _collect_dataset(
    dataset: str,
    dataloader,
    specs: Sequence[CheckpointSpec],
    configs: Mapping[str, PlanConfig],
    models: Mapping[str, torch.nn.Module],
    graph: BpeCtcTrainingGraphCompiler,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    cut_ids: List[str] = []
    seen = set()

    for batch_index, batch in enumerate(dataloader):
        feature = batch["inputs"].to(device)
        supervisions = batch["supervisions"]
        raw_lengths = supervisions["num_frames"].to(device)
        texts: Sequence[str] = supervisions["text"]
        cuts = supervisions["cut"]
        token_ids = graph.texts_to_ids(list(texts))
        reference_ctc: Dict[str, torch.Tensor] = {}
        reference_plan: Dict[str, torch.Tensor] = {}

        for spec_index, spec in enumerate(specs):
            config = configs[spec.name]
            model = models[spec.name]
            with torch.inference_mode():
                (
                    gated,
                    nonblank,
                    alpha,
                    encoder_lengths,
                    _,
                    _,
                ) = model.alignment_forward(feature, supervisions, warmup=1.0)

            for utterance_index, ids in enumerate(token_ids):
                cut_id = cuts[utterance_index].id
                if cut_id not in seen:
                    seen.add(cut_id)
                    cut_ids.append(cut_id)
                length = int(encoder_lengths[utterance_index].item())
                labels = torch.tensor(ids, device=device, dtype=torch.long)
                if length <= 0 or labels.numel() == 0:
                    continue
                gated_i = gated[utterance_index, :length]
                nonblank_i = nonblank[utterance_index, :length]
                alpha_i = alpha[utterance_index, :length]
                acoustic_i = _aligned_acoustic_features(
                    feature[utterance_index],
                    raw_num_frames=int(raw_lengths[utterance_index].item()),
                    encoder_length=length,
                    subsampling_factor=config.subsampling_factor,
                )
                with torch.inference_mode():
                    ctc = _compute_ctc_state_occupancy(gated_i, labels, blank_id=0)
                    ot_plan = _reconstruct_plan(
                        nonblank_i,
                        alpha_i,
                        labels,
                        acoustic_i,
                        config,
                        lambda_gw=0.0,
                    )
                    if math.isclose(config.lambda_gw, 0.0, abs_tol=1.0e-12):
                        fgw_plan = ot_plan
                    else:
                        fgw_plan = _reconstruct_plan(
                            nonblank_i,
                            alpha_i,
                            labels,
                            acoustic_i,
                            config,
                            lambda_gw=config.lambda_gw,
                        )

                row: Dict[str, Any] = {
                    "dataset": dataset,
                    "cut_id": cut_id,
                    "model": spec.name,
                    "lambda_gw": spec.lambda_gw,
                    "num_frames": length,
                    "num_tokens": int(labels.numel()),
                }
                _add_prefixed(
                    row,
                    "ot",
                    _alignment_geometry(
                        ot_plan,
                        args.support_relative_threshold,
                        args.diagonal_band_width,
                    ),
                )
                _add_prefixed(
                    row,
                    "fgw",
                    _alignment_geometry(
                        fgw_plan,
                        args.support_relative_threshold,
                        args.diagonal_band_width,
                    ),
                )
                _add_prefixed(
                    row,
                    "ctc",
                    _alignment_geometry(
                        ctc,
                        args.support_relative_threshold,
                        args.diagonal_band_width,
                    ),
                )
                _add_prefixed(
                    row,
                    "direct_fgw_vs_ot",
                    _agreement(
                        fgw_plan,
                        ot_plan,
                        args.support_relative_threshold,
                    ),
                )
                _add_prefixed(
                    row,
                    "fgw_vs_ctc",
                    _agreement(
                        fgw_plan,
                        ctc,
                        args.support_relative_threshold,
                    ),
                )
                _add_prefixed(
                    row,
                    "ot_vs_ctc",
                    _agreement(
                        ot_plan,
                        ctc,
                        args.support_relative_threshold,
                    ),
                )
                row["fgw_minus_ot_diag_mean_abs_dev"] = (
                    row["fgw_diag_mean_abs_dev"] - row["ot_diag_mean_abs_dev"]
                )

                if spec_index == 0:
                    reference_ctc[cut_id] = ctc.detach().cpu()
                    reference_plan[cut_id] = fgw_plan.detach().cpu()
                    cross_plan = {
                        "barycenter_mad": 0.0,
                        "support_iou": 1.0,
                        "total_variation": 0.0,
                    }
                    cross_ctc = dict(cross_plan)
                else:
                    cross_plan = _agreement(
                        fgw_plan,
                        reference_plan[cut_id],
                        args.support_relative_threshold,
                    )
                    cross_ctc = _agreement(
                        ctc,
                        reference_ctc[cut_id],
                        args.support_relative_threshold,
                    )
                _add_prefixed(row, "plan_vs_gw0", cross_plan)
                _add_prefixed(row, "ctc_vs_gw0", cross_ctc)
                row["plan_diag_delta_vs_gw0"] = (
                    row["fgw_diag_mean_abs_dev"]
                    - _alignment_geometry(
                        reference_plan[cut_id],
                        args.support_relative_threshold,
                        args.diagonal_band_width,
                    )["diag_mean_abs_dev"]
                )
                row["ctc_diag_delta_vs_gw0"] = (
                    row["ctc_diag_mean_abs_dev"]
                    - _alignment_geometry(
                        reference_ctc[cut_id],
                        args.support_relative_threshold,
                        args.diagonal_band_width,
                    )["diag_mean_abs_dev"]
                )
                rows.append(row)

            del gated, nonblank, alpha, encoder_lengths

        if (batch_index + 1) % 5 == 0:
            logging.info(
                "%s: processed %d cuts (%d batches)",
                dataset,
                len(seen),
                batch_index + 1,
            )
    return rows, cut_ids


def _metric_keys(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    excluded = {
        "dataset",
        "cut_id",
        "model",
        "lambda_gw",
        "num_frames",
        "num_tokens",
    }
    return [key for key in rows[0] if key not in excluded]


def _bootstrap_mean_ci(
    values: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, float]:
    bootstrap = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _rank(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1)
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = _rank(left)
    right_rank = _rank(right)
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _summarize(
    rows: Sequence[Dict[str, Any]],
    specs: Sequence[CheckpointSpec],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = np.random.default_rng(args.bootstrap_seed)
    metric_keys = _metric_keys(rows)
    output: Dict[str, Any] = {"datasets": {}, "lambda_trends": {}}
    for dataset in [*args.datasets, "combined"]:
        output["datasets"][dataset] = {}
        for spec in specs:
            selected = [
                row
                for row in rows
                if row["model"] == spec.name
                and (dataset == "combined" or row["dataset"] == dataset)
            ]
            info = {
                "lambda_gw": spec.lambda_gw,
                "num_utterances": len(selected),
                "metrics": {},
            }
            indices = rng.integers(
                0,
                len(selected),
                size=(args.bootstrap_samples, len(selected)),
                dtype=np.int32,
            )
            for key in metric_keys:
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                info["metrics"][key] = _bootstrap_mean_ci(values, indices)
            output["datasets"][dataset][spec.name] = info

    lambdas = [spec.lambda_gw for spec in specs]
    combined = output["datasets"]["combined"]
    for key in metric_keys:
        means = [combined[spec.name]["metrics"][key]["mean"] for spec in specs]
        output["lambda_trends"][key] = {
            "spearman_rho": _spearman(lambdas, means),
            "values": dict(zip((spec.name for spec in specs), means)),
        }
    return output


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    combined = summary["datasets"]["combined"]
    lines = [
        "# FGW plan-to-CTC transfer diagnostic",
        "",
        "> Plans use each checkpoint's exact training metadata. Geometry is "
        "computed on the full encoder-frame x transcript-token alignment; no "
        "posterior keep-threshold is applied before plan metrics.",
        "",
        "## Training-matched plan and CTC agreement",
        "",
        "| Model | lambda_gw | FGW diag dev | CTC diag dev | FGW support | "
        "CTC support | FGW/CTC bary MAD | Support IoU | TV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, info in combined.items():
        metric = info["metrics"]
        lines.append(
            f"| {model} | {info['lambda_gw']:g} | "
            f"{metric['fgw_diag_mean_abs_dev']['mean']:.5f} | "
            f"{metric['ctc_diag_mean_abs_dev']['mean']:.5f} | "
            f"{metric['fgw_support_mean_frames']['mean']:.3f} | "
            f"{metric['ctc_support_mean_frames']['mean']:.3f} | "
            f"{metric['fgw_vs_ctc_barycenter_mad']['mean']:.5f} | "
            f"{metric['fgw_vs_ctc_support_iou']['mean']:.4f} | "
            f"{metric['fgw_vs_ctc_total_variation']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Direct structural-cost effect within each checkpoint",
            "",
            "The comparison is the training-matched FGW plan versus a "
            "lambda_gw=0 counterfactual plan from the same model outputs.",
            "",
            "| Model | lambda_gw | Barycenter shift | Support IoU | TV | "
            "FGW-OT diag delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, info in combined.items():
        metric = info["metrics"]
        lines.append(
            f"| {model} | {info['lambda_gw']:g} | "
            f"{metric['direct_fgw_vs_ot_barycenter_mad']['mean']:.5f} | "
            f"{metric['direct_fgw_vs_ot_support_iou']['mean']:.4f} | "
            f"{metric['direct_fgw_vs_ot_total_variation']['mean']:.4f} | "
            f"{metric['fgw_minus_ot_diag_mean_abs_dev']['mean']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-checkpoint shift relative to lambda_gw=0",
            "",
            "| Model | lambda_gw | Plan bary shift | CTC bary shift | "
            "Plan support IoU | CTC support IoU | Plan TV | CTC TV |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, info in combined.items():
        metric = info["metrics"]
        lines.append(
            f"| {model} | {info['lambda_gw']:g} | "
            f"{metric['plan_vs_gw0_barycenter_mad']['mean']:.5f} | "
            f"{metric['ctc_vs_gw0_barycenter_mad']['mean']:.5f} | "
            f"{metric['plan_vs_gw0_support_iou']['mean']:.4f} | "
            f"{metric['ctc_vs_gw0_support_iou']['mean']:.4f} | "
            f"{metric['plan_vs_gw0_total_variation']['mean']:.4f} | "
            f"{metric['ctc_vs_gw0_total_variation']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive lambda trends",
            "",
            "Spearman rho uses five independently trained seed-42 checkpoints "
            "and is descriptive, not a significance test.",
            "",
            "| Metric | rho(lambda, metric) |",
            "|---|---:|",
        ]
    )
    for key in [
        "direct_fgw_vs_ot_barycenter_mad",
        "direct_fgw_vs_ot_support_iou",
        "direct_fgw_vs_ot_total_variation",
        "fgw_diag_mean_abs_dev",
        "ctc_diag_mean_abs_dev",
        "fgw_vs_ctc_barycenter_mad",
        "fgw_vs_ctc_support_iou",
        "plan_vs_gw0_barycenter_mad",
        "ctc_vs_gw0_barycenter_mad",
    ]:
        rho = summary["lambda_trends"][key]["spearman_rho"]
        lines.append(f"| {key} | {rho:+.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = get_parser().parse_args()
    specs = _parse_specs(args.checkpoint)
    if args.max_cuts_per_dataset <= 0:
        raise ValueError("--max-cuts-per-dataset must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(f"{args.output_dir}/log-evaluate-fgw-plan-transfer")
    logging.info("Arguments: %s", args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    lexicon = Lexicon(args.lang_dir)
    num_classes = max(lexicon.tokens) + 1
    graph = BpeCtcTrainingGraphCompiler(
        args.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )

    configs: Dict[str, PlanConfig] = {}
    checkpoint_metadata: Dict[str, Any] = {}
    models: Dict[str, torch.nn.Module] = {}
    for spec in specs:
        config, metadata = _plan_config(spec, args.epoch)
        configs[spec.name] = config
        checkpoint_metadata[spec.name] = {
            "exp_dir": str(spec.exp_dir),
            "epoch": args.epoch,
            "avg": args.avg,
            "plan_config": asdict(config),
            "batch_idx_train": metadata.get("batch_idx_train"),
            "seed": metadata.get("seed"),
        }
        models[spec.name] = _build_vfta_model(
            spec.exp_dir, args, num_classes, device
        )

    all_rows: List[Dict[str, Any]] = []
    cut_manifest: Dict[str, List[str]] = {}
    for dataset in args.datasets:
        dataset_args = copy.copy(args)
        dataset_args.dataset = dataset
        dataset_args.max_cuts = args.max_cuts_per_dataset
        dataloader = _load_eval_dataloader(dataset_args)
        rows, cut_ids = _collect_dataset(
            dataset,
            dataloader,
            specs,
            configs,
            models,
            graph,
            device,
            args,
        )
        all_rows.extend(rows)
        cut_manifest[dataset] = cut_ids
        logging.info("Completed %s with %d cuts", dataset, len(cut_ids))

    expected = args.max_cuts_per_dataset * len(args.datasets)
    actual = sum(len(cut_ids) for cut_ids in cut_manifest.values())
    if actual != expected:
        raise RuntimeError(f"Expected {expected} cuts, evaluated {actual}")
    expected_rows = expected * len(specs)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, got {len(all_rows)}")

    summary = _summarize(all_rows, specs, args)
    summary["configuration"] = {
        "datasets": args.datasets,
        "max_cuts_per_dataset": args.max_cuts_per_dataset,
        "support_relative_threshold": args.support_relative_threshold,
        "diagonal_band_width": args.diagonal_band_width,
        "use_averaged_model": args.use_averaged_model,
        "checkpoint_metadata": checkpoint_metadata,
    }
    _write_csv(all_rows, args.output_dir / "utterance_metrics.csv")
    (args.output_dir / "cut_ids.json").write_text(
        json.dumps(cut_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, args.output_dir / "summary.md")
    logging.info("Saved FGW plan-transfer diagnostic under %s", args.output_dir)


if __name__ == "__main__":
    main()
