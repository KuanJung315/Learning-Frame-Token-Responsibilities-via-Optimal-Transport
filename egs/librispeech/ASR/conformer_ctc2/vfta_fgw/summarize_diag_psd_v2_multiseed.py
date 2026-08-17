#!/usr/bin/env python3
"""Aggregate the preregistered learned-vs-frozen diag-PSD-v2 experiment."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


TIMIT_METRICS = {
    "PBE_ms": ("PBE_ms_utterance_macro", "PBE_ms"),
    "WBE_ms": ("WBE_ms_utterance_macro", "WBE_ms"),
    "phone_onset_ms": (
        "phone_onset_mae_ms_utterance_macro",
        "phone_onset_mae_ms",
    ),
    "phone_offset_ms": (
        "phone_offset_mae_ms_utterance_macro",
        "phone_offset_mae_ms",
    ),
    "word_onset_ms": (
        "word_onset_mae_ms_utterance_macro",
        "word_onset_mae_ms",
    ),
    "word_offset_ms": (
        "word_offset_mae_ms_utterance_macro",
        "word_offset_mae_ms",
    ),
    "plan_ctc_w1_frames": ("plan_ctc_w1_frames", "plan_ctc_w1_frames"),
    "plan_ctc_barycenter_mae_frames": (
        "plan_ctc_barycenter_mae_frames",
        "plan_ctc_barycenter_mae_frames",
    ),
    "plan_ctc_support_iou": ("plan_ctc_support_iou", "plan_ctc_support_iou"),
    "plan_diagonal_deviation": (
        "plan_diagonal_deviation",
        "plan_diagonal_deviation",
    ),
    "ctc_diagonal_deviation": (
        "ctc_diagonal_deviation",
        "ctc_diagonal_deviation",
    ),
}


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> Dict[str, dict]:
    rows = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows[row["cut_id"]] = row
    return rows


def finite_differences(
    frozen_path: Path, learned_path: Path, key: str
) -> np.ndarray:
    frozen = load_jsonl(frozen_path)
    learned = load_jsonl(learned_path)
    values = []
    for cut_id in sorted(frozen.keys() & learned.keys()):
        a = frozen[cut_id].get(key)
        b = learned[cut_id].get(key)
        if a is None or b is None:
            continue
        if math.isfinite(float(a)) and math.isfinite(float(b)):
            values.append(float(b) - float(a))
    return np.asarray(values, dtype=np.float64)


def paired_bootstrap(values: np.ndarray, seed: int, samples: int) -> dict:
    if values.size == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        boot[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "n": int(values.size),
    }


def seed_paths(root: Path, seed: int) -> dict:
    if seed == 42:
        return {
            "mfa": root
            / "diag_psd_v2_matched_frozen_eval/e15_avg5/mfa/summary.json",
            "timit_frozen": root
            / "diag_psd_v2_matched_frozen_eval/e15_avg5/timit_dev_frozen",
            "timit_learned": root
            / "diag_psd_v2_averaged_full/e15_avg5/timit_dev_diag_psd_v2",
            "rho": root / "diag_psd_v2_averaged_full/e15_avg5/rho_sweep/summary.json",
        }
    base = root / f"diag_psd_v2_multiseed_eval/seed{seed}/e15_avg5"
    return {
        "mfa": base / "mfa/summary.json",
        "timit_frozen": base / "timit_dev_frozen",
        "timit_learned": base / "timit_dev_learned",
        "rho": base / "rho_sweep/summary.json",
    }


def collect_seed(root: Path, seed: int, bootstrap_samples: int) -> dict:
    paths = seed_paths(root, seed)
    required = [
        paths["mfa"],
        paths["timit_frozen"] / "summary.json",
        paths["timit_learned"] / "summary.json",
        paths["rho"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"seed": seed, "complete": False, "missing": missing}

    mfa = load_json(paths["mfa"])
    candidate = next(name for name in mfa["models"] if name != "baseline")
    combined = mfa["datasets"]["combined"]
    frozen_mfa = combined["baseline"]["utterance_macro_metrics"]
    learned_mfa = combined[candidate]["utterance_macro_metrics"]
    paired_mfa = mfa["pairwise"]["combined"]["metrics"]
    paired_geometry = mfa["geometry"]["combined"]["paired"]

    frozen_timit = load_json(paths["timit_frozen"] / "summary.json")["DEV"]
    learned_timit = load_json(paths["timit_learned"] / "summary.json")["DEV"]
    timit = {
        name: {
            "frozen": float(frozen_timit[summary_key]),
            "learned": float(learned_timit[summary_key]),
            "delta": float(learned_timit[summary_key] - frozen_timit[summary_key]),
        }
        for name, (summary_key, _) in TIMIT_METRICS.items()
    }

    frozen_details = paths["timit_frozen"] / "dev-details.jsonl"
    learned_details = paths["timit_learned"] / "dev-details.jsonl"
    for metric_name, (_, detail_key) in TIMIT_METRICS.items():
        values = finite_differences(frozen_details, learned_details, detail_key)
        timit[metric_name]["paired_bootstrap"] = paired_bootstrap(
            values, seed=10000 + seed, samples=bootstrap_samples
        )

    rho = load_json(paths["rho"])
    rho_combined = rho["datasets"]["combined"]
    endpoint = rho_combined["endpoint_paired"]
    spearman = rho_combined["mean_curve_spearman_rho"]

    return {
        "seed": seed,
        "complete": True,
        "libri": {
            "utterances": int(combined["baseline"]["num_utterances"]),
            "words": int(combined["baseline"]["num_words"]),
            "WBE_ms": {
                "frozen": frozen_mfa["boundary_abs_error_ms"]["mean"],
                "learned": learned_mfa["boundary_abs_error_ms"]["mean"],
                **paired_mfa["boundary_abs_error_ms"],
            },
            "word_iou": {
                "frozen": frozen_mfa["word_iou"]["mean"],
                "learned": learned_mfa["word_iou"]["mean"],
                **paired_mfa["word_iou"],
            },
            "geometry_delta": {
                key: value["candidate_minus_baseline"]
                for key, value in paired_geometry.items()
            },
        },
        "timit": timit,
        "rho": {
            "metric_spectrum": rho["metric_spectrum"],
            "endpoint_delta": {
                key: value["rho_max_minus_rho_min"] for key, value in endpoint.items()
            },
            "spearman": spearman,
        },
    }


def mean_sd(values: Iterable[float], lower_is_better: bool = True) -> dict:
    values = list(values)
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "num_seeds": len(values),
        "num_improved": sum(
            value < 0 if lower_is_better else value > 0 for value in values
        ),
    }


def aggregate(records: List[dict]) -> dict:
    complete = [record for record in records if record["complete"]]
    if not complete:
        return {"num_complete_seeds": 0}
    return {
        "num_complete_seeds": len(complete),
        "seeds": [record["seed"] for record in complete],
        "libri_WBE_delta_ms": mean_sd(
            record["libri"]["WBE_ms"]["candidate_minus_baseline"]
            for record in complete
        ),
        "libri_word_iou_delta": mean_sd(
            (
                record["libri"]["word_iou"]["candidate_minus_baseline"]
                for record in complete
            ),
            lower_is_better=False,
        ),
        "timit_PBE_delta_ms": mean_sd(
            record["timit"]["PBE_ms"]["delta"] for record in complete
        ),
        "timit_WBE_delta_ms": mean_sd(
            record["timit"]["WBE_ms"]["delta"] for record in complete
        ),
        "rho_plan_ctc_W1_endpoint_delta_frames": mean_sd(
            record["rho"]["endpoint_delta"]["plan_ctc_w1_frames"]
            for record in complete
        ),
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_markdown(records: List[dict], aggregate_summary: dict) -> str:
    complete = [record for record in records if record["complete"]]
    pending = [record for record in records if not record["complete"]]
    lines = [
        "# Locked diag-PSD-v2 multi-seed results",
        "",
        "Protocol: LibriSpeech clean-100 phone model, common epoch-10 checkpoint per seed,",
        "paired learned/frozen continuations through epoch 15, and fixed epoch 10→15",
        "interval averaging. TIMIT is zero-shot and excludes SA utterances.",
        "",
        "## LibriSpeech MFA DEV",
        "",
        "| Seed | Frozen WBE | Learned WBE | Δ WBE (95% CI) | Frozen IoU | Learned IoU | Δ IoU (95% CI) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in complete:
        wbe = record["libri"]["WBE_ms"]
        iou = record["libri"]["word_iou"]
        lines.append(
            f"| {record['seed']} | {fmt(wbe['frozen'], 2)} | {fmt(wbe['learned'], 2)} | "
            f"{wbe['candidate_minus_baseline']:+.3f} [{wbe['ci95_low']:+.3f}, {wbe['ci95_high']:+.3f}] | "
            f"{fmt(iou['frozen'], 4)} | {fmt(iou['learned'], 4)} | "
            f"{iou['candidate_minus_baseline']:+.5f} [{iou['ci95_low']:+.5f}, {iou['ci95_high']:+.5f}] |"
        )

    lines.extend(
        [
            "",
            "## TIMIT DEV zero-shot",
            "",
            "| Seed | Frozen PBE | Learned PBE | Δ PBE (95% CI) | Frozen WBE | Learned WBE | Δ WBE (95% CI) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in complete:
        pbe = record["timit"]["PBE_ms"]
        wbe = record["timit"]["WBE_ms"]
        pbe_ci = pbe["paired_bootstrap"]["ci95"]
        wbe_ci = wbe["paired_bootstrap"]["ci95"]
        lines.append(
            f"| {record['seed']} | {fmt(pbe['frozen'])} | {fmt(pbe['learned'])} | "
            f"{pbe['delta']:+.3f} [{pbe_ci[0]:+.3f}, {pbe_ci[1]:+.3f}] | "
            f"{fmt(wbe['frozen'])} | {fmt(wbe['learned'])} | "
            f"{wbe['delta']:+.3f} [{wbe_ci[0]:+.3f}, {wbe_ci[1]:+.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Rho geometry endpoint (rho 1 minus rho 0)",
            "",
            "| Seed | Distance rel. Fro | Plan TV | Barycenter shift | Support IoU change | Plan–CTC W1 change | W1 Spearman |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in complete:
        endpoint = record["rho"]["endpoint_delta"]
        spearman = record["rho"]["spearman"]["plan_ctc_w1_frames"]
        lines.append(
            f"| {record['seed']} | {endpoint['distance_relative_fro_to_rho0']:.6f} | "
            f"{endpoint['plan_tv_to_rho0']:.6f} | "
            f"{endpoint['plan_barycenter_mae_to_rho0']:.4f} | "
            f"{endpoint['plan_support_iou_to_rho0']:+.6f} | "
            f"{endpoint['plan_ctc_w1_frames']:+.6f} | {spearman:+.2f} |"
        )

    if len(complete) > 1:
        lines.extend(
            [
                "",
                "## Across-seed descriptive summary",
                "",
                "Mean ± sample SD is descriptive with only three seeds; utterance-level paired CIs remain in the per-seed rows.",
                "",
                "| Quantity | Mean Δ | Sample SD | Seeds improved |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, key in (
            ("Libri WBE (ms)", "libri_WBE_delta_ms"),
            ("TIMIT PBE (ms)", "timit_PBE_delta_ms"),
            ("TIMIT WBE (ms)", "timit_WBE_delta_ms"),
            ("rho endpoint plan–CTC W1 (frames)", "rho_plan_ctc_W1_endpoint_delta_frames"),
        ):
            value = aggregate_summary[key]
            lines.append(
                f"| {label} | {value['mean']:+.4f} | {value['sample_sd']:.4f} | "
                f"{value['num_improved']}/{value['num_seeds']} |"
            )

    if pending:
        lines.extend(["", "## Pending seeds", ""])
        for record in pending:
            lines.append(f"- Seed {record['seed']}: waiting for {len(record['missing'])} result files.")

    lines.extend(
        [
            "",
            "MFA is an automatic pseudo-reference. These are DEV results and must not be labeled as final TEST results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("conformer_ctc2/vfta_fgw"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/vfta_fgw/diag_psd_v2_multiseed_eval"),
    )
    args = parser.parse_args()

    records = [
        collect_seed(args.root, seed, args.bootstrap_samples) for seed in args.seeds
    ]
    aggregate_summary = aggregate(records)
    payload = {"records": records, "aggregate": aggregate_summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "summary.md").write_text(
        render_markdown(records, aggregate_summary)
    )


if __name__ == "__main__":
    main()
