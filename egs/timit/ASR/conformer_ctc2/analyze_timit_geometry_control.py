#!/usr/bin/env python3
"""Aggregate the TIMIT FGW plan-to-CTC geometry-control diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


CORE_METRICS = (
    "direct_plan_vs_ot__barycenter_mad",
    "direct_plan_vs_ot__support_iou",
    "direct_plan_vs_ot__total_variation",
    "direct_plan_minus_ot_diag_mean_abs_dev",
    "plan_vs_gw0__barycenter_mad",
    "plan_vs_gw0__support_iou",
    "plan_vs_gw0__total_variation",
    "ctc_prior_vs_gw0__barycenter_mad",
    "ctc_prior_vs_gw0__support_iou",
    "ctc_prior_vs_gw0__total_variation",
    "matched_plan__diag_mean_abs_dev",
    "ctc_prior__diag_mean_abs_dev",
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument(
        "--run", action="append", required=True, metavar="SEED=SUMMARY_JSON"
    )
    parser.add_argument("--candidate-lambda", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_run(raw: str) -> Tuple[int, Path]:
    seed, path = raw.split("=", maxsplit=1)
    return int(seed), Path(path)


def _rank(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1)
        index = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = _rank(left)
    y = _rank(right)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def find_lambda_model(models: Dict[str, Any], target: float) -> str:
    matches = [
        name
        for name, info in models.items()
        if np.isclose(float(info["lambda_gw"]), target)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one lambda={target} model, found {matches}")
    return matches[0]


def mean_metric(info: Dict[str, Any], metric: str) -> float:
    return float(info["metrics"][metric]["mean"])


def analyze_sweep(summary: Dict[str, Any]) -> Dict[str, Any]:
    output = {}
    for split, models in summary["splits"].items():
        ordered = sorted(models.items(), key=lambda item: float(item[1]["lambda_gw"]))
        lambdas = [float(info["lambda_gw"]) for _, info in ordered]
        rows = []
        for name, info in ordered:
            row = {"model": name, "lambda_gw": float(info["lambda_gw"])}
            for metric in CORE_METRICS:
                row[metric] = mean_metric(info, metric)
            rows.append(row)
        correlations = {
            "lambda_vs_plan_diag_spearman": spearman(
                lambdas, [row["matched_plan__diag_mean_abs_dev"] for row in rows]
            ),
            "lambda_vs_ctc_diag_spearman": spearman(
                lambdas, [row["ctc_prior__diag_mean_abs_dev"] for row in rows]
            ),
            "lambda_vs_direct_plan_bary_spearman": spearman(
                lambdas,
                [row["direct_plan_vs_ot__barycenter_mad"] for row in rows],
            ),
            "plan_vs_ctc_cross_lambda_bary_spearman": spearman(
                [row["plan_vs_gw0__barycenter_mad"] for row in rows],
                [row["ctc_prior_vs_gw0__barycenter_mad"] for row in rows],
            ),
            "plan_vs_ctc_absolute_diag_spearman": spearman(
                [row["matched_plan__diag_mean_abs_dev"] for row in rows],
                [row["ctc_prior__diag_mean_abs_dev"] for row in rows],
            ),
        }
        output[split] = {"rows": rows, "correlations": correlations}
    return output


def analyze_multiseed(
    runs: Sequence[Tuple[int, Dict[str, Any]]], candidate_lambda: float
) -> Dict[str, Any]:
    output = {}
    for split in ("dev", "test"):
        seed_rows = {}
        for seed, summary in runs:
            models = summary["splits"][split]
            reference = models[find_lambda_model(models, 0.0)]
            candidate = models[find_lambda_model(models, candidate_lambda)]
            row = {metric: mean_metric(candidate, metric) for metric in CORE_METRICS}
            plan_bary = row["plan_vs_gw0__barycenter_mad"]
            plan_tv = row["plan_vs_gw0__total_variation"]
            plan_support_change = 1.0 - row["plan_vs_gw0__support_iou"]
            row.update(
                {
                    "plan_diag_response": row["matched_plan__diag_mean_abs_dev"]
                    - mean_metric(reference, "matched_plan__diag_mean_abs_dev"),
                    "ctc_diag_response": row["ctc_prior__diag_mean_abs_dev"]
                    - mean_metric(reference, "ctc_prior__diag_mean_abs_dev"),
                    "ctc_to_plan_bary_ratio": row[
                        "ctc_prior_vs_gw0__barycenter_mad"
                    ]
                    / max(plan_bary, 1.0e-12),
                    "ctc_to_plan_tv_ratio": row["ctc_prior_vs_gw0__total_variation"]
                    / max(plan_tv, 1.0e-12),
                    "ctc_to_plan_support_change_ratio": (
                        1.0 - row["ctc_prior_vs_gw0__support_iou"]
                    )
                    / max(plan_support_change, 1.0e-12),
                    "plan_change_ci95_excludes_zero": (
                        float(
                            candidate["metrics"][
                                "plan_vs_gw0__barycenter_mad"
                            ]["ci95_low"]
                        )
                        > 0.0
                    ),
                    "ctc_change_ci95_excludes_zero": (
                        float(
                            candidate["metrics"][
                                "ctc_prior_vs_gw0__barycenter_mad"
                            ]["ci95_low"]
                        )
                        > 0.0
                    ),
                }
            )
            seed_rows[str(seed)] = row
        metric_names = list(next(iter(seed_rows.values())))
        aggregate = {}
        for metric in metric_names:
            values = [row[metric] for row in seed_rows.values()]
            if isinstance(values[0], bool):
                aggregate[metric] = {"all_seeds": bool(all(values)), "values": values}
            else:
                aggregate[metric] = {
                    "mean": float(np.mean(values)),
                    "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }
        output[split] = {"seeds": seed_rows, "aggregate": aggregate}
    return output


def write_markdown(analysis: Dict[str, Any], path: Path) -> None:
    lines = ["# TIMIT FGW geometry-control analysis", ""]
    candidate_lambda = float(analysis["candidate_lambda"])
    for split in ("dev", "test"):
        sweep = analysis["sweep"][split]
        lines += [
            f"## {split} lambda sweep (seed 42)",
            "",
            "| lambda | direct plan/OT bary | direct support IoU | direct diag delta | plan-vs-0 bary | CTC-vs-0 bary | plan diag | CTC diag |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sweep["rows"]:
            lines.append(
                f"| {row['lambda_gw']:.2f} | "
                f"{row['direct_plan_vs_ot__barycenter_mad']:.5f} | "
                f"{row['direct_plan_vs_ot__support_iou']:.4f} | "
                f"{row['direct_plan_minus_ot_diag_mean_abs_dev']:+.5f} | "
                f"{row['plan_vs_gw0__barycenter_mad']:.5f} | "
                f"{row['ctc_prior_vs_gw0__barycenter_mad']:.5f} | "
                f"{row['matched_plan__diag_mean_abs_dev']:.5f} | "
                f"{row['ctc_prior__diag_mean_abs_dev']:.5f} |"
            )
        lines += ["", "Spearman correlations:", ""]
        for key, value in sweep["correlations"].items():
            lines.append(f"- {key}: {value:+.4f}")
        aggregate = analysis["multiseed"][split]["aggregate"]

        def show(metric: str, digits: int = 5) -> str:
            value = aggregate[metric]
            return f"{value['mean']:.{digits}f} +/- {value['seed_std']:.{digits}f}"

        lines += [
            "",
            f"## {split} lambda={candidate_lambda:g} multi-seed transfer",
            "",
            "| plan-vs-0 bary | CTC-vs-0 bary | plan diag response | CTC diag response | CTC/plan bary ratio | CTC/plan TV ratio |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {show('plan_vs_gw0__barycenter_mad')} | "
            f"{show('ctc_prior_vs_gw0__barycenter_mad')} | "
            f"{show('plan_diag_response')} | {show('ctc_diag_response')} | "
            f"{show('ctc_to_plan_bary_ratio', 3)} | "
            f"{show('ctc_to_plan_tv_ratio', 3)} |",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = get_parser().parse_args()
    sweep = json.loads(args.sweep_summary.read_text())
    runs = [
        (seed, json.loads(path.read_text()))
        for seed, path in (parse_run(raw) for raw in args.run)
    ]
    analysis = {
        "candidate_lambda": args.candidate_lambda,
        "sweep": analyze_sweep(sweep),
        "multiseed": analyze_multiseed(runs, args.candidate_lambda),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown(analysis, args.output_dir / "analysis.md")


if __name__ == "__main__":
    main()
