#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class SystemSpec:
    name: str
    seed: int
    role: str
    directory: Path


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    candidate: str
    reference: str


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=SEED=ROLE=DIR",
        help="ROLE is baseline or vi; DIR contains alignment/ and decode/.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="NAME=CANDIDATE=REFERENCE",
        help=(
            "Compute paired candidate-minus-reference deltas by cut and seed. "
            "Negative is better for error rates; positive is better for F1/accuracy."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_specs(raw_specs: Sequence[str]) -> List[SystemSpec]:
    specs = []
    keys = set()
    for raw in raw_specs:
        parts = raw.split("=", maxsplit=3)
        if len(parts) != 4:
            raise ValueError(f"Expected NAME=SEED=ROLE=DIR, got {raw}")
        name, seed, role, directory = parts
        if role not in ("baseline", "vi"):
            raise ValueError(f"Invalid role {role}")
        key = (name, int(seed))
        if key in keys:
            raise ValueError(f"Duplicate system/seed: {key}")
        keys.add(key)
        specs.append(SystemSpec(name, int(seed), role, Path(directory)))
    return specs


def parse_comparisons(raw_specs: Sequence[str]) -> List[ComparisonSpec]:
    specs = []
    names = set()
    for raw in raw_specs:
        parts = raw.split("=", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"Expected NAME=CANDIDATE=REFERENCE, got {raw}")
        name, candidate, reference = parts
        if name in names:
            raise ValueError(f"Duplicate comparison name: {name}")
        names.add(name)
        specs.append(ComparisonSpec(name, candidate, reference))
    return specs


def load_rows(spec: SystemSpec, split: str) -> List[Dict[str, str]]:
    path = spec.directory / "alignment" / f"{split}-alignment-metrics.csv"
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def _ratio(rows, numerator: str, denominator: str) -> float:
    den = sum(float(row[denominator]) for row in rows)
    return sum(float(row[numerator]) for row in rows) / max(den, 1.0)


def aggregate_rows(rows: Sequence[Dict[str, str]], role: str) -> Dict[str, float]:
    prefix = "baseline_ctc_forced" if role == "baseline" else "vi_ctc_forced"
    metrics = {
        "phone_pbe_ms": _ratio(
            rows,
            f"{prefix}__phone_boundary_abs_sum_ms",
            f"{prefix}__phone_count",
        ),
        "phone_onset_mae_ms": _ratio(
            rows,
            f"{prefix}__phone_onset_abs_sum_ms",
            f"{prefix}__phone_count",
        ),
        "phone_offset_mae_ms": _ratio(
            rows,
            f"{prefix}__phone_offset_abs_sum_ms",
            f"{prefix}__phone_count",
        ),
        "pdur_mae_ms": _ratio(
            rows,
            f"{prefix}__phone_duration_abs_sum_ms",
            f"{prefix}__phone_count",
        ),
        "pdur_bias_ms": _ratio(
            rows,
            f"{prefix}__phone_duration_signed_sum_ms",
            f"{prefix}__phone_count",
        ),
        "predicted_pdur_ms": _ratio(
            rows,
            f"{prefix}__phone_predicted_duration_sum_ms",
            f"{prefix}__phone_count",
        ),
        "gold_pdur_ms": _ratio(
            rows,
            f"{prefix}__phone_gold_duration_sum_ms",
            f"{prefix}__phone_count",
        ),
        "nonsil_phone_pbe_ms": _ratio(
            rows,
            f"{prefix}__phone_nonsil_boundary_abs_sum_ms",
            f"{prefix}__phone_nonsil_count",
        ),
        "word_wbe_ms": _ratio(
            rows,
            f"{prefix}__word_boundary_abs_sum_ms",
            f"{prefix}__word_count",
        ),
        "word_onset_mae_ms": _ratio(
            rows,
            f"{prefix}__word_onset_abs_sum_ms",
            f"{prefix}__word_count",
        ),
        "word_offset_mae_ms": _ratio(
            rows,
            f"{prefix}__word_offset_abs_sum_ms",
            f"{prefix}__word_count",
        ),
        "wdur_mae_ms": _ratio(
            rows,
            f"{prefix}__word_duration_abs_sum_ms",
            f"{prefix}__word_count",
        ),
        "wdur_bias_ms": _ratio(
            rows,
            f"{prefix}__word_duration_signed_sum_ms",
            f"{prefix}__word_count",
        ),
        "predicted_wdur_ms": _ratio(
            rows,
            f"{prefix}__word_predicted_duration_sum_ms",
            f"{prefix}__word_count",
        ),
        "gold_wdur_ms": _ratio(
            rows,
            f"{prefix}__word_gold_duration_sum_ms",
            f"{prefix}__word_count",
        ),
        "center_mae_ms": _ratio(
            rows,
            f"{prefix}__center_abs_error_sum_ms",
            f"{prefix}__center_count",
        ),
        "frame_accuracy": _ratio(
            rows,
            f"{prefix}__frame_correct",
            f"{prefix}__frame_count",
        ),
    }
    for tolerance in (10, 20, 30, 50):
        tag = f"{tolerance}ms"
        matches = sum(float(row[f"{prefix}__boundary_matches_{tag}"]) for row in rows)
        predicted = sum(
            float(row[f"{prefix}__boundary_pred_count_{tag}"]) for row in rows
        )
        gold = sum(float(row[f"{prefix}__boundary_gold_count_{tag}"]) for row in rows)
        precision = matches / max(predicted, 1.0)
        recall = matches / max(gold, 1.0)
        metrics[f"boundary_f1_{tag}"] = (
            2.0 * precision * recall / max(precision + recall, 1.0e-8)
        )
    if role == "vi":
        geometry_keys = (
            "matched_plan_vs_ot__barycenter_mad",
            "matched_plan_vs_ot__support_iou",
            "matched_plan_vs_ot__total_variation",
            "matched_plan_minus_ot_diag_mean_abs_dev",
            "matched_plan_vs_ctc__barycenter_mad",
            "matched_plan_vs_ctc__support_iou",
        )
        for key in geometry_keys:
            metrics[key] = float(np.mean([float(row[key]) for row in rows]))
    return metrics


def bootstrap(
    rows: Sequence[Dict[str, str]], role: str, samples: int, rng
) -> Dict[str, Dict[str, float]]:
    point = aggregate_rows(rows, role)
    values: Dict[str, List[float]] = {key: [] for key in point}
    for _ in range(samples):
        indices = rng.integers(0, len(rows), size=len(rows))
        selected = [rows[int(index)] for index in indices]
        result = aggregate_rows(selected, role)
        for key, value in result.items():
            values[key].append(value)
    output = {}
    for key, mean in point.items():
        low, high = np.quantile(values[key], [0.025, 0.975])
        output[key] = {
            "mean": float(mean),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return output


def paired_bootstrap(
    candidate_rows: Sequence[Dict[str, str]],
    candidate_role: str,
    reference_rows: Sequence[Dict[str, str]],
    reference_role: str,
    samples: int,
    rng,
) -> Dict[str, Dict[str, float]]:
    candidate_by_cut = {row["cut_id"]: row for row in candidate_rows}
    reference_by_cut = {row["cut_id"]: row for row in reference_rows}
    cut_ids = sorted(set(candidate_by_cut) & set(reference_by_cut))
    if not cut_ids:
        raise ValueError("No common cut_id values in paired comparison")

    def delta(selected_ids: Sequence[str]) -> Dict[str, float]:
        candidate = aggregate_rows(
            [candidate_by_cut[cut_id] for cut_id in selected_ids], candidate_role
        )
        reference = aggregate_rows(
            [reference_by_cut[cut_id] for cut_id in selected_ids], reference_role
        )
        return {
            key: candidate[key] - reference[key]
            for key in sorted(set(candidate) & set(reference))
        }

    point = delta(cut_ids)
    values: Dict[str, List[float]] = {key: [] for key in point}
    for _ in range(samples):
        indices = rng.integers(0, len(cut_ids), size=len(cut_ids))
        selected_ids = [cut_ids[int(index)] for index in indices]
        result = delta(selected_ids)
        for key, value in result.items():
            values[key].append(value)

    output = {}
    for key, mean in point.items():
        low, high = np.quantile(values[key], [0.025, 0.975])
        output[key] = {
            "mean": float(mean),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    output["num_common_utterances"] = {"mean": len(cut_ids)}
    return output


def load_per(spec: SystemSpec, split: str) -> float:
    path = spec.directory / "decode" / "per-summary.json"
    values = json.loads(path.read_text())
    key = "baseline_greedy" if spec.role == "baseline" else "vi_prior_greedy"
    return float(values[split][key])


def summarize(
    specs: Sequence[SystemSpec],
    comparisons: Sequence[ComparisonSpec],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    output: Dict[str, Any] = {"runs": {}, "systems": {}, "comparisons": {}}
    for spec in specs:
        run_key = f"{spec.name}/seed{spec.seed}"
        output["runs"][run_key] = {"role": spec.role, "seed": spec.seed, "splits": {}}
        for split in ("dev", "test"):
            result = bootstrap(load_rows(spec, split), spec.role, samples, rng)
            result["per"] = {"mean": load_per(spec, split)}
            output["runs"][run_key]["splits"][split] = result

    for name in sorted({spec.name for spec in specs}):
        selected = [spec for spec in specs if spec.name == name]
        output["systems"][name] = {"num_seeds": len(selected), "splits": {}}
        for split in ("dev", "test"):
            metric_keys = output["runs"][f"{name}/seed{selected[0].seed}"]["splits"][split]
            aggregate = {}
            for metric in metric_keys:
                points = [
                    output["runs"][f"{name}/seed{spec.seed}"]["splits"][split][metric]["mean"]
                    for spec in selected
                ]
                aggregate[metric] = {
                    "mean": float(np.mean(points)),
                    "seed_std": float(np.std(points, ddof=1)) if len(points) > 1 else 0.0,
                    "values": dict(zip((str(spec.seed) for spec in selected), points)),
                }
            output["systems"][name]["splits"][split] = aggregate

    by_name_seed = {(spec.name, spec.seed): spec for spec in specs}
    known_names = {spec.name for spec in specs}
    for comparison in comparisons:
        if comparison.candidate not in known_names:
            raise ValueError(f"Unknown candidate system: {comparison.candidate}")
        if comparison.reference not in known_names:
            raise ValueError(f"Unknown reference system: {comparison.reference}")
        candidate_seeds = {
            spec.seed for spec in specs if spec.name == comparison.candidate
        }
        reference_seeds = {
            spec.seed for spec in specs if spec.name == comparison.reference
        }
        common_seeds = sorted(candidate_seeds & reference_seeds)
        if not common_seeds:
            raise ValueError(f"No common seeds for comparison {comparison.name}")
        comparison_output: Dict[str, Any] = {
            "candidate": comparison.candidate,
            "reference": comparison.reference,
            "num_seeds": len(common_seeds),
            "runs": {},
            "splits": {},
        }
        for split in ("dev", "test"):
            per_seed = {}
            for run_seed in common_seeds:
                candidate = by_name_seed[(comparison.candidate, run_seed)]
                reference = by_name_seed[(comparison.reference, run_seed)]
                result = paired_bootstrap(
                    load_rows(candidate, split),
                    candidate.role,
                    load_rows(reference, split),
                    reference.role,
                    samples,
                    rng,
                )
                result["per"] = {
                    "mean": load_per(candidate, split) - load_per(reference, split)
                }
                per_seed[str(run_seed)] = result
            metric_keys = per_seed[str(common_seeds[0])]
            aggregate = {}
            for metric in metric_keys:
                points = [per_seed[str(run_seed)][metric]["mean"] for run_seed in common_seeds]
                aggregate[metric] = {
                    "mean": float(np.mean(points)),
                    "seed_std": (
                        float(np.std(points, ddof=1)) if len(points) > 1 else 0.0
                    ),
                    "values": dict(zip((str(value) for value in common_seeds), points)),
                }
            comparison_output["runs"][split] = per_seed
            comparison_output["splits"][split] = aggregate
        output["comparisons"][comparison.name] = comparison_output
    return output


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    systems = summary["systems"]
    lines = ["# TIMIT VFTA-FGW experiment summary", ""]
    for split in ("dev", "test"):
        lines += [f"## {split}", "", "| System | Seeds | PBE ms | nonsil PBE | WBE ms | PDUR MAE | Pred/Gold PDUR | WDur MAE | Pred/Gold WDur | F1@20 | PER |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, info in systems.items():
            values = info["splits"][split]
            def show(key: str, digits: int = 2) -> str:
                value = values[key]
                return f"{value['mean']:.{digits}f}±{value['seed_std']:.{digits}f}"
            lines.append(
                f"| {name} | {info['num_seeds']} | {show('phone_pbe_ms')} | "
                f"{show('nonsil_phone_pbe_ms')} | {show('word_wbe_ms')} | "
                f"{show('pdur_mae_ms')} | {show('predicted_pdur_ms')}/{show('gold_pdur_ms')} | "
                f"{show('wdur_mae_ms')} | {show('predicted_wdur_ms')}/{show('gold_wdur_ms')} | "
                f"{show('boundary_f1_20ms', 4)} | {show('per')} |"
            )
        lines.append("")
    if summary["comparisons"]:
        lines += [
            "## Paired candidate-minus-reference deltas",
            "",
            "Negative is better for PBE/WBE/PDUR/PER; positive is better for F1.",
            "",
        ]
        for split in ("dev", "test"):
            lines += [
                f"### {split}",
                "",
                "| Comparison | Seeds | ΔPBE ms | ΔWBE ms | ΔPDUR ms | ΔF1@20 | ΔPER |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for name, info in summary["comparisons"].items():
                values = info["splits"][split]

                def show_delta(key: str, digits: int = 2) -> str:
                    value = values[key]
                    return f"{value['mean']:+.{digits}f}±{value['seed_std']:.{digits}f}"

                lines.append(
                    f"| {name} | {info['num_seeds']} | {show_delta('phone_pbe_ms')} | "
                    f"{show_delta('word_wbe_ms')} | {show_delta('pdur_mae_ms')} | "
                    f"{show_delta('boundary_f1_20ms', 4)} | {show_delta('per')} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(
        parse_specs(args.system),
        parse_comparisons(args.comparison),
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown(summary, args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
