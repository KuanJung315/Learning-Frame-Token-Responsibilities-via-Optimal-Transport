#!/usr/bin/env python3
"""Combine per-checkpoint MFA diagnostic summaries into one paper-style table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


ORDER = {
    "baseline": 0,
    "label_prior_official_a0p3": 1,
    "label_prior_stable_a0p03": 2,
    "adamer": 3,
    "vfta_gw0": 4,
    "vfta_gw0p05": 5,
    "vfta_gw0p1": 6,
    "vfta_gw0p2": 7,
    "vfta_gw0p3": 8,
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("conformer_ctc2/mfa_alignment_diagnostic"),
    )
    parser.add_argument("--tag", type=str, default="full_dev")
    return parser


def _row(
    model: str,
    info: Dict[str, Any],
    delta: Dict[str, Any] | None,
    configuration: Dict[str, Any],
) -> Dict[str, Any]:
    macro = info["utterance_macro_metrics"]
    micro = info["metrics"]
    boundary_delta = (
        delta["metrics"]["boundary_abs_error_ms"] if delta is not None else None
    )
    iou_delta = delta["metrics"]["word_iou"] if delta is not None else None
    return {
        "model": model,
        "num_utterances": info["num_utterances"],
        "num_words": info["num_words"],
        "wbe_macro_ms": macro["boundary_abs_error_ms"]["mean"],
        "onset_macro_ms": macro["start_abs_error_ms"]["mean"],
        "offset_macro_ms": macro["end_abs_error_ms"]["mean"],
        "pred_wdur_macro_ms": macro["pred_duration_ms"]["mean"],
        "ref_wdur_macro_ms": macro["ref_duration_ms"]["mean"],
        "duration_mae_macro_ms": macro["duration_abs_error_ms"]["mean"],
        "center_mae_macro_ms": macro["center_abs_error_ms"]["mean"],
        "word_iou_macro": macro["word_iou"]["mean"],
        "within_80ms_micro": micro["boundary_within_80ms"]["mean"],
        "wbe_delta_vs_baseline_ms": (
            boundary_delta["candidate_minus_baseline"]
            if boundary_delta is not None
            else 0.0
        ),
        "wbe_delta_ci95_low": (
            boundary_delta["ci95_low"] if boundary_delta is not None else 0.0
        ),
        "wbe_delta_ci95_high": (
            boundary_delta["ci95_high"] if boundary_delta is not None else 0.0
        ),
        "word_iou_delta_vs_baseline": (
            iou_delta["candidate_minus_baseline"] if iou_delta is not None else 0.0
        ),
        "candidate_kind": configuration.get("candidate_kind", "baseline"),
        "candidate_exp_dir": configuration.get("candidate_exp_dir", ""),
        "candidate_epoch": configuration.get("candidate_epoch", ""),
        "candidate_avg": configuration.get("candidate_avg", ""),
    }


def main() -> None:
    args = get_parser().parse_args()
    paths = sorted(args.input_root.glob(f"*_{args.tag}/summary.json"))
    if not paths:
        raise FileNotFoundError(
            f"No summaries matched {args.input_root}/*_{args.tag}/summary.json"
        )

    rows: List[Dict[str, Any]] = []
    baseline_added = False
    sources = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        combined = summary["datasets"]["combined"]
        candidates = [name for name in combined if name != "baseline"]
        if len(candidates) != 1:
            raise ValueError(f"Expected one candidate in {path}, got {candidates}")
        candidate = candidates[0]
        if not baseline_added:
            rows.append(
                _row(
                    "baseline",
                    combined["baseline"],
                    None,
                    {
                        "candidate_exp_dir": summary["configuration"][
                            "baseline_exp_dir"
                        ],
                        "candidate_epoch": summary["configuration"][
                            "baseline_epoch"
                        ],
                        "candidate_avg": summary["configuration"]["baseline_avg"],
                    },
                )
            )
            baseline_added = True
        rows.append(
            _row(
                candidate,
                combined[candidate],
                summary["pairwise"]["combined"],
                summary["configuration"],
            )
        )
        sources.append(str(path))

    rows.sort(key=lambda row: (ORDER.get(row["model"], 999), row["model"]))
    args.input_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.input_root / f"diagnostic_summary_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.input_root / f"diagnostic_summary_{args.tag}.json"
    json_path.write_text(
        json.dumps({"rows": rows, "sources": sources}, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# MFA alignment diagnostic sweep",
        "",
        "> WBE is the utterance-macro word boundary error used to match the "
        "Label-Prior CTC paper's aggregation. MFA is an automatic "
        "pseudo-reference.",
        "",
        "| Model | WBE | Onset | Offset | Pred. WDUR | Ref. WDUR | "
        "Duration MAE | Word IoU | WBE delta (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['wbe_macro_ms']:.2f} | "
            f"{row['onset_macro_ms']:.2f} | {row['offset_macro_ms']:.2f} | "
            f"{row['pred_wdur_macro_ms']:.2f} | "
            f"{row['ref_wdur_macro_ms']:.2f} | "
            f"{row['duration_mae_macro_ms']:.2f} | "
            f"{row['word_iou_macro']:.4f} | "
            f"{row['wbe_delta_vs_baseline_ms']:+.2f} "
            f"[{row['wbe_delta_ci95_low']:+.2f}, "
            f"{row['wbe_delta_ci95_high']:+.2f}] |"
        )
    md_path = args.input_root / f"diagnostic_summary_{args.tag}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
