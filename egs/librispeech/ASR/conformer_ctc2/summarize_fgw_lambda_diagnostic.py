#!/usr/bin/env python3
"""Join MFA boundary quality and posterior geometry for the FGW lambda sweep."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


LAMBDA_MODELS = [
    (0.0, "vfta_gw0"),
    (0.05, "vfta_gw0p05"),
    (0.1, "vfta_gw0p1"),
    (0.2, "vfta_gw0p2"),
    (0.3, "vfta_gw0p3"),
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mfa-summary",
        type=Path,
        default=Path(
            "conformer_ctc2/mfa_alignment_diagnostic/"
            "diagnostic_summary_full_dev.json"
        ),
    )
    parser.add_argument(
        "--geometry-summary",
        type=Path,
        default=Path(
            "conformer_ctc2/alignment_geometry_diagnostic_2000/summary.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("conformer_ctc2/fgw_lambda_alignment_diagnostic"),
    )
    return parser


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = 0.5 * (index + end - 1)
        for position in range(index, end):
            ranks[order[position]] = average_rank
        index = end
    return ranks


def _pearson(left: List[float], right: List[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_norm = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_norm = sum((y - right_mean) ** 2 for y in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return numerator / (left_norm * right_norm)


def _spearman(left: List[float], right: List[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def _bootstrap_delta(
    candidate: Dict[Any, float],
    baseline: Dict[Any, float],
    rng: np.random.Generator,
    num_samples: int = 2000,
) -> Dict[str, float]:
    keys = sorted(candidate.keys() & baseline.keys())
    deltas = np.asarray(
        [candidate[key] - baseline[key] for key in keys], dtype=np.float64
    )
    bootstrap = np.empty(num_samples, dtype=np.float64)
    for index in range(num_samples):
        sample = rng.integers(0, len(deltas), size=len(deltas))
        bootstrap[index] = deltas[sample].mean()
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "num_pairs": len(keys),
        "mean": float(deltas.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _read_utterance_wbe(path: Path, model: str) -> Dict[Any, float]:
    accumulated: Dict[Any, List[float]] = defaultdict(lambda: [0.0, 0.0])
    with path.open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["model"] != model:
                continue
            key = (row["dataset"], row["cut_id"])
            accumulated[key][0] += float(row["start_abs_error_ms"])
            accumulated[key][0] += float(row["end_abs_error_ms"])
            accumulated[key][1] += 2.0
    return {key: total / count for key, (total, count) in accumulated.items()}


def _read_geometry(path: Path) -> Dict[str, Dict[str, Dict[Any, float]]]:
    output: Dict[str, Dict[str, Dict[Any, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    fields = [
        "mean_nonblank_prob",
        "mean_frame_entropy",
        "spike_width_mean",
        "ctc_diag_mean_abs_dev",
    ]
    with path.open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["mode"] != "native":
                continue
            if row["model"] not in {model for _, model in LAMBDA_MODELS}:
                continue
            key = (row["dataset"], row["cut_id"])
            for field in fields:
                output[row["model"]][field][key] = float(row[field])
    return output


def main() -> None:
    args = get_parser().parse_args()
    mfa = json.loads(args.mfa_summary.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry_summary.read_text(encoding="utf-8"))
    mfa_rows = {row["model"]: row for row in mfa["rows"]}
    model_geometry = geometry["models"]["combined"]["native"]

    tag = args.mfa_summary.stem.removeprefix("diagnostic_summary_")
    utterance_wbe = {
        model: _read_utterance_wbe(
            args.mfa_summary.parent / f"{model}_{tag}" / "word_metrics.csv",
            model,
        )
        for _, model in LAMBDA_MODELS
    }
    cut_geometry = _read_geometry(
        args.geometry_summary.parent / "utterance_metrics.csv"
    )
    rng = np.random.default_rng(20260720)
    paired_vs_gw0: Dict[str, Any] = {}
    for _, model in LAMBDA_MODELS:
        paired_vs_gw0[model] = {
            "wbe_macro_ms": _bootstrap_delta(
                utterance_wbe[model], utterance_wbe["vfta_gw0"], rng
            ),
            "geometry": {},
        }
        for field in cut_geometry["vfta_gw0"]:
            paired_vs_gw0[model]["geometry"][field] = _bootstrap_delta(
                cut_geometry[model][field],
                cut_geometry["vfta_gw0"][field],
                rng,
            )

    rows: List[Dict[str, Any]] = []
    for lambda_gw, model in LAMBDA_MODELS:
        boundary = mfa_rows[model]
        posterior = model_geometry[model]
        metrics = posterior["metrics"]
        rows.append(
            {
                "lambda_gw": lambda_gw,
                "model": model,
                "wbe_macro_ms": boundary["wbe_macro_ms"],
                "wbe_delta_vs_baseline_ms": boundary[
                    "wbe_delta_vs_baseline_ms"
                ],
                "word_iou_macro": boundary["word_iou_macro"],
                "pred_wdur_macro_ms": boundary["pred_wdur_macro_ms"],
                "greedy_wer": 100.0 * posterior["greedy_errors"]["wer"],
                "mean_nonblank_prob": metrics["mean_nonblank_prob"]["mean"],
                "mean_frame_entropy": metrics["mean_frame_entropy"]["mean"],
                "spike_width_mean": metrics["spike_width_mean"]["mean"],
                "ctc_diag_mean_abs_dev": metrics[
                    "ctc_diag_mean_abs_dev"
                ]["mean"],
                "ctc_offdiag_mass": metrics["ctc_offdiag_mass"]["mean"],
                "ctc_bary_jitter": metrics["ctc_bary_jitter"]["mean"],
            }
        )

    lambdas = [row["lambda_gw"] for row in rows]
    correlation_fields = [
        "wbe_macro_ms",
        "word_iou_macro",
        "pred_wdur_macro_ms",
        "greedy_wer",
        "mean_nonblank_prob",
        "mean_frame_entropy",
        "spike_width_mean",
        "ctc_diag_mean_abs_dev",
        "ctc_offdiag_mass",
        "ctc_bary_jitter",
    ]
    correlations = {
        field: _spearman(lambdas, [float(row[field]) for row in rows])
        for field in correlation_fields
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "spearman_rho_vs_lambda": correlations,
                "sources": {
                    "mfa": str(args.mfa_summary),
                    "geometry": str(args.geometry_summary),
                },
                "paired_vs_gw0": paired_vs_gw0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# FGW lambda alignment diagnostic",
        "",
        "> MFA WBE is an utterance-macro pseudo-reference metric. Posterior "
        "geometry uses 2,000 matched cuts at the native operating point.",
        "",
        "| lambda_gw | WBE | Word IoU | Pred. WDUR | Greedy WER | "
        "Nonblank | Entropy | Spike width | CTC diag dev | Offdiag | Jitter |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['lambda_gw']:g} | {row['wbe_macro_ms']:.2f} | "
            f"{row['word_iou_macro']:.4f} | "
            f"{row['pred_wdur_macro_ms']:.2f} | "
            f"{row['greedy_wer']:.2f} | "
            f"{row['mean_nonblank_prob']:.4f} | "
            f"{row['mean_frame_entropy']:.4f} | "
            f"{row['spike_width_mean']:.4f} | "
            f"{row['ctc_diag_mean_abs_dev']:.5f} | "
            f"{row['ctc_offdiag_mass']:.5f} | "
            f"{row['ctc_bary_jitter']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Rank trend across the five trained checkpoints",
            "",
            "Spearman rho is descriptive only (five lambda values, one seed); "
            "it is not a significance test.",
            "",
            "| Metric | rho(lambda, metric) |",
            "|---|---:|",
        ]
    )
    for field in correlation_fields:
        lines.append(f"| {field} | {correlations[field]:+.3f} |")
    lines.extend(
        [
            "",
            "## Paired deltas versus lambda_gw=0",
            "",
            "| lambda_gw | WBE delta (95% CI) | Nonblank delta | "
            "Entropy delta | Spike-width delta | Diag-dev delta |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for lambda_gw, model in LAMBDA_MODELS:
        paired = paired_vs_gw0[model]
        wbe = paired["wbe_macro_ms"]
        geom = paired["geometry"]
        cells = []
        for field in [
            "mean_nonblank_prob",
            "mean_frame_entropy",
            "spike_width_mean",
            "ctc_diag_mean_abs_dev",
        ]:
            metric = geom[field]
            cells.append(
                f"{metric['mean']:+.5f} "
                f"[{metric['ci95_low']:+.5f}, {metric['ci95_high']:+.5f}]"
            )
        lines.append(
            f"| {lambda_gw:g} | {wbe['mean']:+.2f} "
            f"[{wbe['ci95_low']:+.2f}, {wbe['ci95_high']:+.2f}] | "
            + " | ".join(cells)
            + " |"
        )
    md_path = args.output_prefix.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
