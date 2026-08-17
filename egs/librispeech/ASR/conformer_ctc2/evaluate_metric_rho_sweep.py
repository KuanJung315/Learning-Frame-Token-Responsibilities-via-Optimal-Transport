#!/usr/bin/env python3
"""Diagnose learned structural-metric sensitivity without retraining.

The encoder, CTC posterior, blank gate, and learned metric parameters all come
from one checkpoint and are evaluated once.  Only the interpolation coefficient

    D_rho = (1-rho) D_cosine + rho D_learned

changes while reconstructing FGW plans.  This separates metric/plan
sensitivity from encoder co-adaptation and plan-to-CTC transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent
FGW_DIR = SCRIPT_DIR / "vfta_fgw"
for path in (SCRIPT_DIR, ASR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(FGW_DIR) not in sys.path:
    sys.path.append(str(FGW_DIR))

from asr_datamodule import LibriSpeechAsrDataModule
from ctc_plan_consistency import ctc_token_occupancy_batched
from evaluate_alignment_metrics import _load_eval_dataloader
from evaluate_mfa_alignment import (
    _reference_word_spans,
    librispeech_utterance_id,
    load_mfa_manifest,
)
from evaluate_mfa_phone_alignment import (
    _build_model,
    _column_normalize,
    _encoder_aligned_raw_features,
    _geometry,
    _input_order_transcripts,
    _model_outputs,
    _normalized_reference_words,
    _padded_phone_targets,
    _reconstruct_plan,
)
from word_phone_graph_compiler import WordPhoneCtcTrainingGraphCompiler

from icefall.lexicon import Lexicon
from icefall.utils import setup_logger


SENSITIVITY_METRICS = (
    "distance_relative_fro_to_rho0",
    "distance_mean_abs_relative_to_rho0",
    "distance_cosine_to_rho0",
    "plan_tv_to_rho0",
    "plan_relative_fro_to_rho0",
    "plan_barycenter_mae_to_rho0",
    "plan_support_iou_to_rho0",
    "plan_diagonal_shift_from_rho0",
    "plan_ctc_w1_frames",
    "plan_ctc_barycenter_mae_frames",
    "plan_ctc_support_iou",
    "plan_diagonal_deviation",
    "ctc_diagonal_deviation",
)


def _distance_sensitivity(
    reference: torch.Tensor, candidate: torch.Tensor
) -> Dict[str, float]:
    """Compare two symmetric frame-distance matrices off the diagonal."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Distance matrices must have the same rank-2 shape")
    if reference.size(0) != reference.size(1):
        raise ValueError("Distance matrices must be square")
    if reference.size(0) <= 1:
        return {
            "distance_relative_fro_to_rho0": 0.0,
            "distance_mean_abs_relative_to_rho0": 0.0,
            "distance_cosine_to_rho0": 1.0,
        }
    mask = ~torch.eye(
        reference.size(0), dtype=torch.bool, device=reference.device
    )
    ref = reference.detach().float()[mask]
    cur = candidate.detach().float()[mask]
    delta = cur - ref
    eps = 1.0e-12
    cosine = torch.dot(ref, cur) / (
        ref.norm().clamp_min(eps) * cur.norm().clamp_min(eps)
    )
    return {
        "distance_relative_fro_to_rho0": float(
            delta.norm() / ref.norm().clamp_min(eps)
        ),
        "distance_mean_abs_relative_to_rho0": float(
            delta.abs().mean() / ref.abs().mean().clamp_min(eps)
        ),
        "distance_cosine_to_rho0": float(cosine),
    }


def _plan_diagonal_deviation(plan: torch.Tensor) -> float:
    plan = _column_normalize(plan)
    num_frames, num_tokens = plan.shape
    if num_frames <= 1 or num_tokens <= 1:
        return 0.0
    time_position = torch.linspace(0, 1, num_frames).unsqueeze(1)
    token_position = torch.linspace(0, 1, num_tokens).unsqueeze(0)
    return float(
        (plan * (time_position - token_position).abs()).sum()
        / max(num_tokens, 1)
    )


def _plan_sensitivity(
    reference: torch.Tensor, candidate: torch.Tensor
) -> Dict[str, float]:
    """Compare token-conditional plans while factoring out column mass."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Plans must have the same rank-2 shape")
    ref = _column_normalize(reference)
    cur = _column_normalize(candidate)
    num_frames, num_tokens = ref.shape
    frame = torch.arange(num_frames, dtype=ref.dtype).unsqueeze(1)
    ref_barycenter = (ref * frame).sum(dim=0)
    cur_barycenter = (cur * frame).sum(dim=0)
    ref_support = ref >= 0.1 * ref.max(dim=0, keepdim=True).values
    cur_support = cur >= 0.1 * cur.max(dim=0, keepdim=True).values
    support_union = (ref_support | cur_support).sum().clamp_min(1)
    support_intersection = (ref_support & cur_support).sum()
    eps = 1.0e-12
    return {
        "plan_tv_to_rho0": float(
            0.5 * (cur - ref).abs().sum() / max(num_tokens, 1)
        ),
        "plan_relative_fro_to_rho0": float(
            (cur - ref).norm() / ref.norm().clamp_min(eps)
        ),
        "plan_barycenter_mae_to_rho0": float(
            (cur_barycenter - ref_barycenter).abs().mean()
        ),
        "plan_support_iou_to_rho0": float(
            support_intersection / support_union
        ),
        "plan_diagonal_shift_from_rho0": (
            _plan_diagonal_deviation(cur) - _plan_diagonal_deviation(ref)
        ),
    }


def _describe(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "std": float(array.std()),
    }


def _ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_rank = _ranks(x)
    y_rank = _ranks(y)
    if float(x_rank.std()) == 0.0 or float(y_rank.std()) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No rho-sweep rows were produced")
    datasets = sorted({str(row["dataset"]) for row in rows})
    rhos = sorted({float(row["rho"]) for row in rows})
    output: Dict[str, Any] = {}
    for dataset in datasets + ["combined"]:
        selected = [
            row
            for row in rows
            if dataset == "combined" or str(row["dataset"]) == dataset
        ]
        by_rho: Dict[str, Any] = {}
        mean_curves: Dict[str, List[float]] = {
            metric: [] for metric in SENSITIVITY_METRICS
        }
        for rho in rhos:
            rho_rows = [row for row in selected if float(row["rho"]) == rho]
            metrics = {
                metric: _describe([float(row[metric]) for row in rho_rows])
                for metric in SENSITIVITY_METRICS
            }
            by_rho[f"{rho:g}"] = {
                "num_utterances": len(rho_rows),
                "metrics": metrics,
            }
            for metric in SENSITIVITY_METRICS:
                mean_curves[metric].append(metrics[metric]["mean"])

        endpoint: Dict[str, Any] = {}
        by_cut_rho: Dict[str, Dict[float, Mapping[str, Any]]] = defaultdict(dict)
        for row in selected:
            by_cut_rho[str(row["cut_id"])][float(row["rho"])] = row
        rho0, rho1 = rhos[0], rhos[-1]
        for metric in SENSITIVITY_METRICS:
            deltas = [
                float(per_rho[rho1][metric]) - float(per_rho[rho0][metric])
                for per_rho in by_cut_rho.values()
                if rho0 in per_rho and rho1 in per_rho
            ]
            mean = float(np.mean(deltas))
            std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
            half_width = 1.96 * std / math.sqrt(max(len(deltas), 1))
            endpoint[metric] = {
                "rho_max_minus_rho_min": mean,
                "ci95": [mean - half_width, mean + half_width],
                "num_paired_utterances": len(deltas),
            }
        output[dataset] = {
            "rhos": by_rho,
            "mean_curve_spearman_rho": {
                metric: _spearman(rhos, values)
                for metric, values in mean_curves.items()
            },
            "endpoint_paired": endpoint,
        }
    return output


def _metric_spectrum(model) -> Dict[str, Any]:
    metric = getattr(model, "structural_metric", None)
    if metric is None:
        raise ValueError("Checkpoint does not contain a learned structural metric")
    weight = metric.effective_projection_weight(rho=1.0).detach().float().cpu()
    gram = weight.transpose(0, 1) @ weight
    eigenvalues = torch.linalg.eigvalsh(gram)
    positive = eigenvalues.clamp_min(1.0e-12)
    probability = positive / positive.sum()
    effective_rank = torch.exp(-(probability * probability.log()).sum())
    identity = torch.eye(weight.size(0))
    return {
        "projection_delta_rms": float((weight - identity).square().mean().sqrt()),
        "gram_identity_error_rms": float((gram - identity).square().mean().sqrt()),
        "eigenvalue_min": float(eigenvalues.min()),
        "eigenvalue_median": float(eigenvalues.median()),
        "eigenvalue_max": float(eigenvalues.max()),
        "condition_number_clamped": float(positive.max() / positive.min()),
        "effective_rank": float(effective_rank),
        "feature_dim": int(weight.size(0)),
        "num_eigenvalues_below_0p01": int((eigenvalues < 0.01).sum()),
        "num_eigenvalues_below_0p1": int((eigenvalues < 0.1).sum()),
        "num_eigenvalues_above_2": int((eigenvalues > 2.0).sum()),
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--rhos", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--lang-dir", type=Path, default=Path("data/lang_phone_nostress")
    )
    parser.add_argument(
        "--mfa-dir", type=Path, default=Path("data/librispeech_mfa")
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["dev-clean", "dev-other"]
    )
    parser.add_argument("--max-cuts-per-dataset", type=int, default=0)
    parser.add_argument("--gate", choices=["prior", "posterior"], default="prior")
    parser.add_argument("--prior-logit-bias", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/vfta_fgw/metric_rho_sweep"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--subsampling-factor", type=int, default=4)
    LibriSpeechAsrDataModule.add_arguments(parser)
    return parser


def evaluate(
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    model, params = _build_model(args.checkpoint, args.lang_dir, device)
    metric = getattr(model, "structural_metric", None)
    if metric is None:
        raise ValueError("The rho sweep requires a learned structural metric")
    factor = int(getattr(params, "subsampling_factor", 4))
    if factor != args.subsampling_factor:
        raise ValueError(
            f"checkpoint subsampling={factor}, evaluator={args.subsampling_factor}"
        )
    normalization = str(getattr(params, "metric_normalization", "none"))
    lexicon = Lexicon(args.lang_dir)
    compiler = WordPhoneCtcTrainingGraphCompiler(
        lang_dir=args.lang_dir, lexicon=lexicon, device=device
    )
    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for dataset in args.datasets:
        references = load_mfa_manifest(args.mfa_dir, dataset)
        args.dataset = dataset
        args.max_cuts = args.max_cuts_per_dataset
        dataloader = _load_eval_dataloader(args)
        processed = 0
        for batch_index, batch in enumerate(dataloader):
            features = batch["inputs"].to(device)
            supervisions = batch["supervisions"]
            texts: Sequence[str] = list(supervisions["text"])
            cuts = supervisions["cut"]
            sequence_idx = [
                int(index) for index in supervisions["sequence_idx"].tolist()
            ]
            transcripts = [compiler.expand_text(text) for text in texts]
            input_transcripts = _input_order_transcripts(
                transcripts, sequence_idx, features.size(0)
            )
            targets, target_lengths = _padded_phone_targets(
                input_transcripts, device
            )
            log_probs, output_lens, log_p_nonblank, alpha = _model_outputs(
                model=model,
                features=features,
                supervisions=supervisions,
                targets=targets,
                target_lengths=target_lengths,
                gate=args.gate,
                prior_logit_bias=args.prior_logit_bias,
            )

            for supervision_index, (cut, transcript) in enumerate(
                zip(cuts, transcripts)
            ):
                input_index = sequence_idx[supervision_index]
                cut_id = librispeech_utterance_id(cut)
                record = references.get(cut_id)
                if record is None or not record.get("words"):
                    skipped[dataset].append(
                        {"cut_id": cut_id, "reason": "missing_mfa_words"}
                    )
                    continue
                reference = _reference_word_spans(record)
                if _normalized_reference_words(compiler, reference) != transcript.words:
                    skipped[dataset].append(
                        {"cut_id": cut_id, "reason": "transcript_mismatch"}
                    )
                    continue
                if transcript.oov_words:
                    skipped[dataset].append(
                        {
                            "cut_id": cut_id,
                            "reason": "oov:" + ",".join(transcript.oov_words),
                        }
                    )
                    continue

                try:
                    output_len = int(output_lens[input_index].item())
                    labels = torch.tensor(
                        transcript.phone_ids, dtype=torch.long, device=device
                    )
                    lp = log_probs[input_index, :output_len]
                    occupancy = ctc_token_occupancy_batched(
                        log_probs=lp.unsqueeze(0),
                        labels=labels.unsqueeze(0),
                        frame_lens=torch.tensor(
                            [output_len], dtype=torch.long, device=device
                        ),
                        label_lens=torch.tensor(
                            [len(transcript.phone_ids)],
                            dtype=torch.long,
                            device=device,
                        ),
                        blank_id=0,
                    )[0]
                    raw_num_frames = int(
                        supervisions["num_frames"][supervision_index]
                    )
                    acoustic_features = _encoder_aligned_raw_features(
                        raw_features=features[input_index],
                        raw_num_frames=raw_num_frames,
                        output_len=output_len,
                        subsampling_factor=factor,
                    )
                    plans: Dict[float, torch.Tensor] = {}
                    distances: Dict[float, torch.Tensor] = {}
                    with torch.no_grad():
                        for rho in args.rhos:
                            distance, _ = metric(
                                acoustic_features,
                                rho=rho,
                                normalization=normalization,
                            )
                            distances[rho] = distance.detach()
                            plans[rho] = _reconstruct_plan(
                                log_p_nonblank=log_p_nonblank[
                                    input_index, :output_len
                                ],
                                alpha=alpha[input_index, :output_len],
                                labels=labels,
                                raw_features=features[input_index],
                                raw_num_frames=raw_num_frames,
                                params=params,
                                structural_metric=metric,
                                metric_rho=rho,
                            )
                    rho0 = args.rhos[0]
                    cut_rows = []
                    for rho in args.rhos:
                        cut_rows.append(
                            {
                                "dataset": dataset,
                                "cut_id": cut_id,
                                "rho": rho,
                                "num_frames": output_len,
                                "num_phones": len(transcript.phone_ids),
                                **_distance_sensitivity(
                                    distances[rho0], distances[rho]
                                ),
                                **_plan_sensitivity(plans[rho0], plans[rho]),
                                **_geometry(plans[rho], occupancy),
                            }
                        )
                    rows.extend(cut_rows)
                except (RuntimeError, ValueError) as error:
                    skipped[dataset].append(
                        {"cut_id": cut_id, "reason": str(error)}
                    )
                    continue
                processed += 1

            if (batch_index + 1) % 10 == 0:
                logging.info(
                    "%s: processed %d utterances (%d batches)",
                    dataset,
                    processed,
                    batch_index + 1,
                )
        logging.info(
            "%s complete: processed=%d skipped=%d",
            dataset,
            processed,
            len(skipped[dataset]),
        )
    return rows, dict(skipped), _metric_spectrum(model)


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No rho-sweep rows were produced")
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    combined = summary["datasets"]["combined"]
    lines = [
        "# Same-checkpoint structural-metric rho sweep",
        "",
        "Encoder, CTC posterior, blank gate, and metric parameters are fixed; "
        "only `rho` changes during distance/FGW-plan reconstruction.",
        "",
        "| rho | distance rel. Fro | plan TV | plan barycenter shift | "
        "plan support IoU | plan-CTC W1 | CTC diag. |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rho, data in combined["rhos"].items():
        metric = data["metrics"]
        lines.append(
            f"| {rho} | "
            f"{metric['distance_relative_fro_to_rho0']['mean']:.6f} | "
            f"{metric['plan_tv_to_rho0']['mean']:.6f} | "
            f"{metric['plan_barycenter_mae_to_rho0']['mean']:.6f} | "
            f"{metric['plan_support_iou_to_rho0']['mean']:.6f} | "
            f"{metric['plan_ctc_w1_frames']['mean']:.6f} | "
            f"{metric['ctc_diagonal_deviation']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Mean-curve Spearman correlation with rho",
            "",
        ]
    )
    for metric, correlation in combined["mean_curve_spearman_rho"].items():
        value = "undefined (constant)" if correlation is None else f"{correlation:.6f}"
        lines.append(f"- `{metric}`: {value}")
    lines.extend(
        [
            "",
            "## Metric spectrum",
            "",
            "```json",
            json.dumps(summary["metric_spectrum"], indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = get_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if any(not 0.0 <= rho <= 1.0 for rho in args.rhos):
        raise ValueError("Every rho must lie in [0, 1]")
    args.rhos = sorted(set(float(rho) for rho in args.rhos))
    if not args.rhos or args.rhos[0] != 0.0:
        raise ValueError("The rho sweep must include 0 as its reference")
    if args.rhos[-1] != 1.0:
        raise ValueError("The rho sweep must include 1 as its learned endpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(f"{args.output_dir}/log-rho-sweep")
    logging.info("Arguments: %s", vars(args))

    rows, skipped, metric_spectrum = evaluate(args)
    summary = {
        "configuration": {
            "checkpoint": str(args.checkpoint),
            "rhos": args.rhos,
            "datasets": args.datasets,
            "gate": args.gate,
            "prior_logit_bias": args.prior_logit_bias,
            "metric_only_ablation": True,
            "encoder_and_ctc_fixed": True,
            "selection_filter": "MFA transcript-matched LibriSpeech cuts",
        },
        "metric_spectrum": metric_spectrum,
        "datasets": summarize_rows(rows),
    }
    _write_csv(rows, args.output_dir / "utterance_rho_geometry.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "skipped.json").write_text(
        json.dumps(skipped, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(summary, args.output_dir / "summary.md")
    logging.info("Wrote metric rho sweep to %s", args.output_dir)


if __name__ == "__main__":
    main()
