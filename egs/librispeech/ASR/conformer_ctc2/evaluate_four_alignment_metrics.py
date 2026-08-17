#!/usr/bin/env python3
"""Matched alignment evaluation for baseline, AdaMER, and VFTA variants."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import kaldialign
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ASR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adamer_ctc.model import AdaMERConformer
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from evaluate_alignment_metrics import _checkpoint_metadata, _load_eval_dataloader
from evaluate_vi_alignment_metrics import (
    ConformerVIV2ForAlignment,
    _apply_ctc_nonblank_logit_bias,
    _build_candidate_model,
    _load_state,
)
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.utils import setup_logger, str2bool
from label_prior_ctc.model import LabelPriorConformer
from ot_prior_v2 import vi_ot_loss_v2
from shared_alignment_viz import (
    compute_alignment_quality_metrics,
    compute_alignment_stats,
    compute_plan_agreement_metrics,
)
from train import get_params as get_baseline_params


COMMON_METRICS = (
    "mean_nonblank_prob",
    "mean_frame_entropy",
    "posterior_peakiness",
    "argmax_nonblank_ratio",
    "spike_width_mean",
    "spike_width_max",
    "spike_run_count",
    "keep_ratio",
    "ctc_diag_mean_abs_dev",
    "ctc_offdiag_mass",
    "ctc_bary_jitter",
    "ctc_backward_rate",
    "ctc_support_mean",
    "ctc_support_ratio",
)

PLAN_METRICS = (
    "ot_diag_mean_abs_dev",
    "ot_offdiag_mass",
    "ot_bary_jitter",
    "ot_backward_rate",
    "ot_support_mean",
    "ot_support_ratio",
    "column_entropy",
    "column_entropy_normalized",
    "column_mass_cv",
    "column_max_uniform_ratio",
    "plan_ctc_barycenter_mad",
    "plan_ctc_support_iou",
    "plan_ctc_total_variation",
)

LOWER_IS_BETTER = {
    "greedy_wer",
    "ctc_diag_mean_abs_dev",
    "ctc_offdiag_mass",
    "ctc_bary_jitter",
    "ctc_backward_rate",
    "ot_diag_mean_abs_dev",
    "ot_offdiag_mass",
    "ot_bary_jitter",
    "ot_backward_rate",
    "plan_ctc_barycenter_mad",
    "plan_ctc_total_variation",
}

HIGHER_IS_BETTER = {"plan_ctc_support_iou"}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--baseline-exp-dir", type=Path, required=True)
    parser.add_argument("--adamer-exp-dir", type=Path, required=True)
    parser.add_argument(
        "--label-prior-exp-dir",
        type=Path,
        default=None,
        help="Optional Label-Prior CTC checkpoint for paper-matched diagnostics.",
    )
    parser.add_argument("--label-prior-epoch", type=int, default=30)
    parser.add_argument("--label-prior-avg", type=int, default=10)
    parser.add_argument("--label-prior-alpha", type=float, default=0.3)
    parser.add_argument("--label-prior-floor", type=float, default=math.exp(-12.0))
    parser.add_argument("--acoustic-exp-dir", type=Path, required=True)
    parser.add_argument("--uniform-exp-dir", type=Path, required=True)
    parser.add_argument(
        "--no-ot-exp-dir",
        type=Path,
        default=None,
        help="Optional VFTA checkpoint trained with lambda_ot=0 for internal comparison.",
    )
    parser.add_argument(
        "--extra-vfta",
        action="append",
        default=[],
        metavar="NAME=EXP_DIR=COLUMN=BETA_POS",
        help=(
            "Optional extra VFTA-style checkpoint. COLUMN may be acoustic, "
            "uniform, or none. BETA_POS is used only when recomputing the "
            "OT plan for plan-level diagnostics."
        ),
    )
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--avg", type=int, default=10)
    parser.add_argument("--use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--label-embed-dim", type=int, default=256)
    parser.add_argument("--init-blank-prob", type=float, default=0.35)
    parser.add_argument("--baseline-calibrated-bias", type=float, default=1.5)
    parser.add_argument("--adamer-calibrated-bias", type=float, default=2.0)
    parser.add_argument("--label-prior-calibrated-bias", type=float, default=0.0)
    parser.add_argument("--vfta-calibrated-bias", type=float, default=2.4)
    parser.add_argument(
        "--modes", nargs="+", choices=["native", "calibrated"],
        default=["native", "calibrated"]
    )
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_bpe_500"))
    parser.add_argument(
        "--datasets", nargs="+", default=["dev-clean", "dev-other"]
    )
    parser.add_argument("--max-cuts-per-dataset", type=int, default=1000)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("conformer_ctc2/alignment_eval_4methods_2000")
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260622)
    parser.add_argument("--ot-tau", type=float, default=0.1)
    parser.add_argument("--ot-eps", type=float, default=0.3)
    parser.add_argument("--ot-iters", type=int, default=30)
    parser.add_argument("--ot-beta-pos", type=float, default=1.0)
    parser.add_argument("--alpha-smooth-mix", type=float, default=0.1)
    parser.add_argument("--ot-token-prior-sigma", type=float, default=0.15)
    parser.add_argument("--ot-token-prior-score-temp", type=float, default=1.0)
    parser.add_argument("--ot-token-prior-floor", type=float, default=0.05)
    parser.add_argument("--support-relative-threshold", type=float, default=0.1)
    parser.add_argument("--diagonal-band-width", type=float, default=0.12)
    parser.add_argument("--backward-tol", type=float, default=0.05)
    LibriSpeechAsrDataModule.add_arguments(parser)
    return parser


def _extra_vfta_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    reserved = {"baseline", "adamer", "acoustic_vfta", "uniform_vfta", "no_ot_vfta"}
    specs: List[Dict[str, Any]] = []
    seen = set()
    for raw_spec in args.extra_vfta:
        parts = raw_spec.split("=", maxsplit=3)
        if len(parts) != 4:
            raise ValueError(
                "--extra-vfta must have format NAME=EXP_DIR=COLUMN=BETA_POS, "
                f"got: {raw_spec}"
            )
        name, exp_dir, column_type, beta_pos = parts
        if not name or name in reserved or name in seen:
            raise ValueError(f"Invalid or duplicate extra VFTA name: {name}")
        seen.add(name)
        column_type = column_type.lower()
        if column_type in {"none", "no_plan", "-"}:
            column: str | None = None
        elif column_type in {"acoustic", "uniform"}:
            column = column_type
        else:
            raise ValueError(
                f"Unsupported COLUMN for --extra-vfta {name}: {column_type}"
            )
        specs.append(
            {
                "name": name,
                "exp_dir": Path(exp_dir),
                "column_type": column,
                "beta_pos": float(beta_pos),
            }
        )
    return specs


def _checkpoint_path(exp_dir: Path, epoch: int) -> Path:
    path = exp_dir / f"epoch-{epoch}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _build_standard_model(
    exp_dir: Path,
    model_cls,
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
    epoch: int | None = None,
    avg: int | None = None,
) -> Conformer:
    epoch = args.epoch if epoch is None else epoch
    avg = args.avg if avg is None else avg
    saved = _checkpoint_metadata(_checkpoint_path(exp_dir, epoch))
    params = get_baseline_params()
    params.update(saved)
    if not hasattr(params, "num_decoder_layers"):
        params.num_decoder_layers = args.num_decoder_layers
    model = model_cls(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )
    _load_state(
        model=model,
        exp_dir=exp_dir,
        epoch=epoch,
        avg=avg,
        use_averaged_model=args.use_averaged_model,
        device=device,
    )
    return model


def _build_label_prior_model(
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> LabelPriorConformer:
    if args.label_prior_exp_dir is None:
        raise ValueError("--label-prior-exp-dir is required")
    model = _build_standard_model(
        args.label_prior_exp_dir,
        LabelPriorConformer,
        args,
        num_classes,
        device,
        epoch=args.label_prior_epoch,
        avg=args.label_prior_avg,
    )
    if not bool(model.label_prior_ready.item()):
        raise RuntimeError("Label-prior checkpoint does not contain a ready prior")
    model.apply_prior_in_forward = args.label_prior_alpha > 0.0
    model.decode_alpha = args.label_prior_alpha
    model.decode_floor = args.label_prior_floor
    return model


def _build_vfta_model(
    exp_dir: Path,
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> ConformerVIV2ForAlignment:
    candidate_args = copy.copy(args)
    candidate_args.candidate_exp_dir = exp_dir
    candidate_args.candidate_epoch = args.epoch
    candidate_args.candidate_avg = args.avg
    candidate_args.candidate_use_averaged_model = args.use_averaged_model
    candidate_args.candidate_num_decoder_layers = args.num_decoder_layers
    candidate_args.candidate_prior_logit_bias = 0.0
    return _build_candidate_model(candidate_args, num_classes, device)


def _output_length(num_frames: int) -> int:
    return max(((num_frames - 1) // 2 - 1) // 2, 1)


def _standard_batch_outputs(
    model: Conformer,
    batch: Dict[str, Any],
    device: torch.device,
) -> List[torch.Tensor]:
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    with torch.inference_mode():
        log_probs, _, _ = model(feature, supervisions, warmup=1.0)
    return [
        log_probs[i, : _output_length(int(supervisions["num_frames"][i]))]
        .detach()
        .cpu()
        for i in range(log_probs.size(0))
    ]


def _label_prior_batch_outputs(
    model: LabelPriorConformer,
    batch: Dict[str, Any],
    device: torch.device,
) -> List[torch.Tensor]:
    outputs = _standard_batch_outputs(model, batch, device)
    # The paper's label-prior scores do not sum to one. Per-frame
    # normalization preserves Viterbi/greedy decisions while making entropy,
    # blank probability, and other posterior geometry metrics meaningful.
    return [output.log_softmax(dim=-1) for output in outputs]


def _vfta_batch_outputs(
    model: ConformerVIV2ForAlignment,
    batch: Dict[str, Any],
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    with torch.inference_mode():
        gated, nonblank, alpha, _, _, _ = model.alignment_forward(
            feature, supervisions, warmup=1.0
        )
    outputs = []
    for i in range(gated.size(0)):
        length = _output_length(int(supervisions["num_frames"][i]))
        outputs.append(
            (
                gated[i, :length].detach().cpu(),
                nonblank[i, :length].detach().cpu(),
                alpha[i, :length].detach().cpu(),
            )
        )
    return outputs


def _biased_alpha(alpha: torch.Tensor, bias: float) -> torch.Tensor:
    if bias == 0.0:
        return alpha
    alpha = alpha.float().clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.sigmoid(torch.logit(alpha) + bias)


def _greedy_words(log_probs: torch.Tensor, sp) -> List[str]:
    ids = log_probs.argmax(dim=-1).tolist()
    collapsed = []
    previous = None
    for token in ids:
        if token != previous and token != 0:
            collapsed.append(int(token))
        previous = token
    return sp.decode(collapsed).strip().split() if collapsed else []


def _error_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> Dict[str, int]:
    counts = {"correct": 0, "sub": 0, "del": 0, "ins": 0}
    for ref, hyp in kaldialign.align(reference, hypothesis, "*"):
        if ref == "*":
            counts["ins"] += 1
        elif hyp == "*":
            counts["del"] += 1
        elif ref != hyp:
            counts["sub"] += 1
        else:
            counts["correct"] += 1
    counts["ref_words"] = len(reference)
    counts["errors"] = counts["sub"] + counts["del"] + counts["ins"]
    return counts


def _plan_budget_metrics(plan: torch.Tensor) -> Dict[str, float]:
    tiny = 1.0e-8
    column = plan.detach().float().sum(dim=0)
    column = column / column.sum().clamp_min(tiny)
    entropy = -(column * column.clamp_min(tiny).log()).sum()
    U = max(int(column.numel()), 1)
    normalized = entropy / math.log(U) if U > 1 else entropy.new_tensor(0.0)
    mean = column.mean().clamp_min(tiny)
    cv = column.std(unbiased=False) / mean
    return {
        "column_entropy": float(entropy.item()),
        "column_entropy_normalized": float(normalized.item()),
        "column_mass_cv": float(cv.item()),
        "column_max_uniform_ratio": float((column.max() * U).item()),
    }


def _metric_row(
    dataset: str,
    cut_id: str,
    mode: str,
    model_name: str,
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    pieces: Sequence[str],
    reference_words: Sequence[str],
    sp,
    args: argparse.Namespace,
    ot_plan: torch.Tensor | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    zero_plan = log_probs.new_zeros((log_probs.size(0), labels.numel()))
    stats = compute_alignment_stats(
        log_probs=log_probs,
        labels=labels,
        token_pieces=pieces,
        blank_id=0,
        tau=args.ot_tau,
        eps=args.ot_eps,
        iters=args.ot_iters,
        beta_pos=args.ot_beta_pos,
        support_relative_threshold=args.support_relative_threshold,
        ot_coupling_override=ot_plan if ot_plan is not None else zero_plan,
    )
    quality = compute_alignment_quality_metrics(
        stats,
        diagonal_band_width=args.diagonal_band_width,
        backward_tol=args.backward_tol,
    )
    errors = _error_counts(reference_words, _greedy_words(log_probs, sp))
    row: Dict[str, Any] = {
        "dataset": dataset,
        "cut_id": cut_id,
        "mode": mode,
        "model": model_name,
        "num_tokens": int(labels.numel()),
        **errors,
    }
    row.update({metric: quality[metric] for metric in COMMON_METRICS})

    if ot_plan is None:
        return row, None

    plan_row: Dict[str, Any] = {
        "dataset": dataset,
        "cut_id": cut_id,
        "mode": mode,
        "model": model_name,
    }
    plan_row.update({metric: quality[metric] for metric in PLAN_METRICS if metric in quality})
    plan_row.update(_plan_budget_metrics(ot_plan))
    plan_row.update(
        compute_plan_agreement_metrics(
            stats["ctc_occupancy"],
            ot_plan,
            support_relative_threshold=args.support_relative_threshold,
        )
    )
    return row, plan_row


def _model_biases(args: argparse.Namespace, mode: str) -> Dict[str, float]:
    model_names = ["baseline", "adamer", "acoustic_vfta", "uniform_vfta"]
    if args.label_prior_exp_dir is not None:
        model_names.append("label_prior")
    if args.no_ot_exp_dir is not None:
        model_names.append("no_ot_vfta")
    model_names.extend(spec["name"] for spec in _extra_vfta_specs(args))
    if mode == "native":
        return {name: 0.0 for name in model_names}
    biases = {
        "baseline": args.baseline_calibrated_bias,
        "adamer": args.adamer_calibrated_bias,
        "label_prior": args.label_prior_calibrated_bias,
        "acoustic_vfta": args.vfta_calibrated_bias,
        "uniform_vfta": args.vfta_calibrated_bias,
    }
    if args.no_ot_exp_dir is not None:
        biases["no_ot_vfta"] = args.vfta_calibrated_bias
    for spec in _extra_vfta_specs(args):
        biases[spec["name"]] = args.vfta_calibrated_bias
    return biases


def _collect_dataset(
    dataset: str,
    dl,
    models: Dict[str, torch.nn.Module],
    graph: BpeCtcTrainingGraphCompiler,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    plan_rows: List[Dict[str, Any]] = []
    cut_ids: List[str] = []
    seen = set()
    extra_specs = _extra_vfta_specs(args)

    for batch_idx, batch in enumerate(dl):
        outputs = {
            "baseline": _standard_batch_outputs(models["baseline"], batch, device),
            "adamer": _standard_batch_outputs(models["adamer"], batch, device),
            "acoustic_vfta": _vfta_batch_outputs(
                models["acoustic_vfta"], batch, device
            ),
            "uniform_vfta": _vfta_batch_outputs(models["uniform_vfta"], batch, device),
        }
        if "label_prior" in models:
            outputs["label_prior"] = _label_prior_batch_outputs(
                models["label_prior"], batch, device
            )
        if "no_ot_vfta" in models:
            outputs["no_ot_vfta"] = _vfta_batch_outputs(
                models["no_ot_vfta"], batch, device
            )
        for spec in extra_specs:
            outputs[spec["name"]] = _vfta_batch_outputs(
                models[spec["name"]], batch, device
            )
        texts: Sequence[str] = batch["supervisions"]["text"]
        cuts = batch["supervisions"]["cut"]
        token_ids = graph.texts_to_ids(list(texts))

        for i, text in enumerate(texts):
            cut_id = cuts[i].id
            if cut_id not in seen:
                seen.add(cut_id)
                cut_ids.append(cut_id)
            labels = torch.tensor(token_ids[i], dtype=torch.long)
            pieces = [graph.sp.id_to_piece(token) for token in token_ids[i]]
            reference_words = text.strip().split()

            for mode in args.modes:
                biases = _model_biases(args, mode)
                standard_models = ["baseline", "adamer"]
                if "label_prior" in models:
                    standard_models.append("label_prior")
                for model_name in standard_models:
                    log_probs = _apply_ctc_nonblank_logit_bias(
                        outputs[model_name][i], biases[model_name]
                    )
                    row, _ = _metric_row(
                        dataset, cut_id, mode, model_name, log_probs, labels,
                        pieces, reference_words, graph.sp, args
                    )
                    rows.append(row)

                plan_specs = [
                    {
                        "name": "acoustic_vfta",
                        "column_type": "acoustic",
                        "beta_pos": args.ot_beta_pos,
                    },
                    {
                        "name": "uniform_vfta",
                        "column_type": "uniform",
                        "beta_pos": args.ot_beta_pos,
                    },
                ]
                plan_specs.extend(
                    spec for spec in extra_specs
                    if spec["column_type"] is not None
                )
                for spec in plan_specs:
                    model_name = spec["name"]
                    gated, nonblank, alpha = outputs[model_name][i]
                    bias = biases[model_name]
                    log_probs = _apply_ctc_nonblank_logit_bias(gated, bias)
                    alpha = _biased_alpha(alpha, bias)
                    _, plan = vi_ot_loss_v2(
                        log_p_nonblank=nonblank,
                        alpha=alpha,
                        labels=labels,
                        column_marginal_type=spec["column_type"],
                        alpha_smooth_mix=args.alpha_smooth_mix,
                        token_prior_sigma=args.ot_token_prior_sigma,
                        token_prior_score_temp=args.ot_token_prior_score_temp,
                        token_prior_floor=args.ot_token_prior_floor,
                        eps=args.ot_eps,
                        iters=args.ot_iters,
                        beta_pos=spec["beta_pos"],
                        return_plan=True,
                    )
                    row, plan_row = _metric_row(
                        dataset, cut_id, mode, model_name, log_probs, labels,
                        pieces, reference_words, graph.sp, args, ot_plan=plan
                    )
                    rows.append(row)
                    assert plan_row is not None
                    plan_rows.append(plan_row)

                no_plan_models = []
                if "no_ot_vfta" in models:
                    no_plan_models.append("no_ot_vfta")
                no_plan_models.extend(
                    spec["name"] for spec in extra_specs
                    if spec["column_type"] is None
                )
                for model_name in no_plan_models:
                    gated, _, _ = outputs[model_name][i]
                    log_probs = _apply_ctc_nonblank_logit_bias(
                        gated, biases[model_name]
                    )
                    row, _ = _metric_row(
                        dataset, cut_id, mode, model_name, log_probs, labels,
                        pieces, reference_words, graph.sp, args
                    )
                    rows.append(row)

        if (batch_idx + 1) % 10 == 0:
            logging.info(
                "%s: processed %d unique cuts (%d batches)",
                dataset, len(seen), batch_idx + 1
            )
    return rows, plan_rows, cut_ids


def _bootstrap_indices(n: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(samples, n), dtype=np.int32)


def _mean_ci(values: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _group_rows(
    rows: Iterable[Dict[str, Any]], dataset: str, mode: str, model: str
) -> List[Dict[str, Any]]:
    return [
        row for row in rows
        if (dataset == "combined" or row["dataset"] == dataset)
        and row["mode"] == mode
        and row["model"] == model
    ]


def _summarize(
    rows: List[Dict[str, Any]],
    plan_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = np.random.default_rng(args.bootstrap_seed)
    datasets = list(args.datasets) + ["combined"]
    extra_names = tuple(spec["name"] for spec in _extra_vfta_specs(args))
    preferred_order = (
        "baseline", "label_prior", "adamer", "no_ot_vfta", *extra_names,
        "uniform_vfta", "acoustic_vfta"
    )
    present_models = {row["model"] for row in rows}
    models = tuple(model for model in preferred_order if model in present_models)
    plan_models = tuple(
        model for model in models if any(row["model"] == model for row in plan_rows)
    )
    summary: Dict[str, Any] = {"models": {}, "pairwise": {}, "vfta_plans": {}}

    for dataset in datasets:
        summary["models"][dataset] = {}
        summary["pairwise"][dataset] = {}
        summary["vfta_plans"][dataset] = {}
        for mode in args.modes:
            summary["models"][dataset][mode] = {}
            summary["pairwise"][dataset][mode] = {}
            summary["vfta_plans"][dataset][mode] = {}
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for model in models:
                model_rows = _group_rows(rows, dataset, mode, model)
                grouped[model] = sorted(model_rows, key=lambda row: row["cut_id"])
                n = len(model_rows)
                indices = _bootstrap_indices(n, args.bootstrap_samples, rng)
                info = {"num_utterances": n, "metrics": {}}
                for metric in COMMON_METRICS:
                    values = np.asarray([row[metric] for row in model_rows], dtype=float)
                    info["metrics"][metric] = _mean_ci(values, indices)
                errors = sum(row["errors"] for row in model_rows)
                ref_words = sum(row["ref_words"] for row in model_rows)
                info["greedy_errors"] = {
                    key: sum(row[key] for row in model_rows)
                    for key in ("sub", "del", "ins", "correct")
                }
                info["greedy_errors"].update(
                    {"errors": errors, "ref_words": ref_words,
                     "wer": errors / max(ref_words, 1)}
                )
                summary["models"][dataset][mode][model] = info

            comparisons = [
                ("adamer", "acoustic_vfta", "adamer_minus_acoustic_vfta"),
                ("acoustic_vfta", "uniform_vfta", "acoustic_minus_uniform_vfta"),
                ("baseline", "acoustic_vfta", "baseline_minus_acoustic_vfta"),
            ]
            if "label_prior" in present_models:
                comparisons.extend(
                    [
                        (
                            "label_prior",
                            "acoustic_vfta",
                            "label_prior_minus_acoustic_vfta",
                        ),
                        (
                            "baseline",
                            "label_prior",
                            "baseline_minus_label_prior",
                        ),
                    ]
                )
            if "no_ot_vfta" in present_models:
                comparisons.append(
                    ("acoustic_vfta", "no_ot_vfta", "acoustic_minus_no_ot_vfta")
                )
            for extra_name in extra_names:
                if extra_name in present_models:
                    comparisons.append(
                        (
                            "acoustic_vfta",
                            extra_name,
                            f"acoustic_minus_{extra_name}",
                        )
                    )
            for left, right, name in comparisons:
                if left not in grouped or right not in grouped:
                    continue
                left_rows, right_rows = grouped[left], grouped[right]
                if [row["cut_id"] for row in left_rows] != [
                    row["cut_id"] for row in right_rows
                ]:
                    raise RuntimeError(f"Unpaired cut IDs for {name}")
                indices = _bootstrap_indices(
                    len(left_rows), args.bootstrap_samples, rng
                )
                comparison = {"left": left, "right": right, "metrics": {}}
                for metric in COMMON_METRICS:
                    delta = np.asarray(
                        [l[metric] - r[metric] for l, r in zip(left_rows, right_rows)],
                        dtype=float,
                    )
                    comparison["metrics"][metric] = _mean_ci(delta, indices)
                summary["pairwise"][dataset][mode][name] = comparison

            for model in plan_models:
                model_rows = _group_rows(plan_rows, dataset, mode, model)
                indices = _bootstrap_indices(
                    len(model_rows), args.bootstrap_samples, rng
                )
                info = {"num_utterances": len(model_rows), "metrics": {}}
                for metric in PLAN_METRICS:
                    values = np.asarray([row[metric] for row in model_rows], dtype=float)
                    info["metrics"][metric] = _mean_ci(values, indices)
                summary["vfta_plans"][dataset][mode][model] = info

    summary["directions"] = {
        metric: "lower_is_better" if metric in LOWER_IS_BETTER
        else "higher_is_better" if metric in HIGHER_IS_BETTER
        else "descriptive"
        for metric in set(COMMON_METRICS) | set(PLAN_METRICS) | {"greedy_wer"}
    }
    return summary


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = ["# Matched alignment evaluation", ""]
    for mode in summary["models"]["combined"]:
        lines.extend([
            f"## {mode.title()} operating point", "",
            "| Model | Greedy WER | Nonblank | Entropy | Spike width | CTC diag dev | CTC jitter |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for model, info in summary["models"]["combined"][mode].items():
            m = info["metrics"]
            lines.append(
                f"| {model} | {100 * info['greedy_errors']['wer']:.2f} | "
                f"{m['mean_nonblank_prob']['mean']:.4f} | "
                f"{m['mean_frame_entropy']['mean']:.4f} | "
                f"{m['spike_width_mean']['mean']:.4f} | "
                f"{m['ctc_diag_mean_abs_dev']['mean']:.4f} | "
                f"{m['ctc_bary_jitter']['mean']:.4f} |"
            )
        lines.extend(["", "### VFTA transport plans", "",
            "| Model | Column entropy (norm.) | OT diag dev | OT support | Plan/CTC bary MAD | Support IoU |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for model, info in summary["vfta_plans"]["combined"][mode].items():
            m = info["metrics"]
            lines.append(
                f"| {model} | {m['column_entropy_normalized']['mean']:.4f} | "
                f"{m['ot_diag_mean_abs_dev']['mean']:.4f} | "
                f"{m['ot_support_mean']['mean']:.4f} | "
                f"{m['plan_ctc_barycenter_mad']['mean']:.4f} | "
                f"{m['plan_ctc_support_iou']['mean']:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = get_parser().parse_args()
    extra_specs = _extra_vfta_specs(args)
    if args.max_cuts_per_dataset <= 0:
        raise ValueError("--max-cuts-per-dataset must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(f"{args.output_dir}/log-evaluate-four-alignment")
    logging.info("Arguments: %s", args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    lexicon = Lexicon(args.lang_dir)
    num_classes = max(lexicon.tokens) + 1
    graph = BpeCtcTrainingGraphCompiler(
        args.lang_dir, device=device, sos_token="<sos/eos>", eos_token="<sos/eos>"
    )
    models = {
        "baseline": _build_standard_model(
            args.baseline_exp_dir, Conformer, args, num_classes, device
        ),
        "adamer": _build_standard_model(
            args.adamer_exp_dir, AdaMERConformer, args, num_classes, device
        ),
        "acoustic_vfta": _build_vfta_model(
            args.acoustic_exp_dir, args, num_classes, device
        ),
        "uniform_vfta": _build_vfta_model(
            args.uniform_exp_dir, args, num_classes, device
        ),
    }
    if args.label_prior_exp_dir is not None:
        models["label_prior"] = _build_label_prior_model(
            args, num_classes, device
        )
    if args.no_ot_exp_dir is not None:
        models["no_ot_vfta"] = _build_vfta_model(
            args.no_ot_exp_dir, args, num_classes, device
        )
    for spec in extra_specs:
        models[spec["name"]] = _build_vfta_model(
            spec["exp_dir"], args, num_classes, device
        )

    all_rows: List[Dict[str, Any]] = []
    all_plan_rows: List[Dict[str, Any]] = []
    cut_manifest: Dict[str, List[str]] = {}
    for dataset in args.datasets:
        dataset_args = copy.copy(args)
        dataset_args.dataset = dataset
        dataset_args.max_cuts = args.max_cuts_per_dataset
        dl = _load_eval_dataloader(dataset_args)
        rows, plan_rows, cut_ids = _collect_dataset(
            dataset, dl, models, graph, device, args
        )
        all_rows.extend(rows)
        all_plan_rows.extend(plan_rows)
        cut_manifest[dataset] = cut_ids
        logging.info("Completed %s with %d cuts", dataset, len(cut_ids))

    expected = args.max_cuts_per_dataset * len(args.datasets)
    actual = sum(len(ids) for ids in cut_manifest.values())
    if actual != expected:
        raise RuntimeError(f"Expected {expected} cuts but evaluated {actual}")

    summary = _summarize(all_rows, all_plan_rows, args)
    summary["configuration"] = {
        "epoch": args.epoch,
        "avg": args.avg,
        "use_averaged_model": args.use_averaged_model,
        "label_prior_exp_dir": (
            str(args.label_prior_exp_dir)
            if args.label_prior_exp_dir is not None
            else None
        ),
        "label_prior_epoch": args.label_prior_epoch,
        "label_prior_avg": args.label_prior_avg,
        "label_prior_alpha": args.label_prior_alpha,
        "label_prior_floor": args.label_prior_floor,
        "max_cuts_per_dataset": args.max_cuts_per_dataset,
        "datasets": args.datasets,
    }
    _write_csv(all_rows, args.output_dir / "utterance_metrics.csv")
    _write_csv(all_plan_rows, args.output_dir / "vfta_plan_metrics.csv")
    with open(args.output_dir / "cut_ids.json", "w") as f:
        json.dump(cut_manifest, f, indent=2)
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    _write_markdown(summary, args.output_dir / "summary.md")
    logging.info("Saved complete evaluation under %s", args.output_dir)


if __name__ == "__main__":
    main()
