#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from evaluate_timit_alignment import (
    alignment_geometry,
    ctc_forced_plan,
    plan_agreement,
    true_vi_plan,
)
from timit_eval_common import (
    add_data_arguments,
    build_phone_graph,
    build_vi_model,
    get_split_dataloader,
    resolve_device,
    vi_batch_outputs,
)
from varctc_v2_utils import build_gated_log_probs_v2

from icefall.utils import setup_logger


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    checkpoint: Path
    lambda_gw: float


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH=LAMBDA_GW",
    )
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_phone"))
    parser.add_argument(
        "--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"]
    )
    parser.add_argument("--max-cuts", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/timit_fgw_transfer"),
    )
    add_data_arguments(parser)
    return parser


def parse_specs(raw_specs: Sequence[str]) -> List[CheckpointSpec]:
    specs = []
    names = set()
    for raw in raw_specs:
        parts = raw.split("=", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"Expected NAME=PATH=LAMBDA_GW, got {raw}")
        name, path, lambda_gw = parts
        if not name or name in names:
            raise ValueError(f"Empty or duplicate name: {name!r}")
        names.add(name)
        specs.append(CheckpointSpec(name, Path(path), float(lambda_gw)))
    specs.sort(key=lambda item: item.lambda_gw)
    if not specs or not math.isclose(specs[0].lambda_gw, 0.0, abs_tol=1.0e-9):
        raise ValueError("The first/reference checkpoint must have lambda_gw=0")
    return specs


def add_metrics(row: Dict[str, Any], prefix: str, values: Mapping[str, float]) -> None:
    for key, value in values.items():
        row[f"{prefix}__{key}"] = value


def collect_split(
    split: str,
    dataloader,
    specs: Sequence[CheckpointSpec],
    models: Mapping[str, torch.nn.Module],
    params: Mapping[str, Any],
    graph,
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(dataloader):
        sequence_idx = batch["supervisions"]["sequence_idx"].tolist()
        cuts = batch["supervisions"]["cut"]
        reference: Dict[str, Dict[str, torch.Tensor]] = {}

        for spec_index, spec in enumerate(specs):
            outputs, supervision_ids = vi_batch_outputs(
                models[spec.name], batch, graph=graph, device=device, gate="prior"
            )
            model_params = params[spec.name]
            checkpoint_lambda = float(getattr(model_params, "lambda_gw", 0.0))
            if not math.isclose(
                checkpoint_lambda, spec.lambda_gw, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise ValueError(
                    f"{spec.name}: CLI lambda {spec.lambda_gw} != checkpoint "
                    f"lambda {checkpoint_lambda}"
                )
            prior_mix = min(
                max(float(getattr(model_params, "train_prior_mix", 0.0)), 0.0), 1.0
            )

            for supervision_index, cut in enumerate(cuts):
                input_index = int(sequence_idx[supervision_index])
                output = outputs[input_index]
                labels = torch.tensor(
                    supervision_ids[supervision_index], dtype=torch.long
                )
                raw_features = batch["inputs"][input_index]
                raw_num_frames = int(
                    batch["supervisions"]["num_frames"][supervision_index]
                )
                matched_plan = true_vi_plan(
                    output,
                    labels=labels,
                    vi_params=model_params,
                    gate="train",
                    raw_features=raw_features,
                    raw_num_frames=raw_num_frames,
                )
                counterfactual_ot = true_vi_plan(
                    output,
                    labels=labels,
                    vi_params=model_params,
                    gate="train",
                    raw_features=raw_features,
                    raw_num_frames=raw_num_frames,
                    lambda_gw_override=0.0,
                )
                ctc_prior = ctc_forced_plan(output.log_probs, labels)
                alpha_train = (
                    prior_mix * output.alpha_prior
                    + (1.0 - prior_mix) * output.alpha_post
                )
                train_log_probs = build_gated_log_probs_v2(
                    output.log_p_nonblank.unsqueeze(0), alpha_train.unsqueeze(0)
                )[0]
                ctc_train = ctc_forced_plan(train_log_probs, labels)

                row: Dict[str, Any] = {
                    "split": split,
                    "cut_id": cut.id,
                    "model": spec.name,
                    "lambda_gw": spec.lambda_gw,
                    "num_frames": output.output_len,
                    "num_tokens": int(labels.numel()),
                }
                for prefix, alignment in (
                    ("matched_plan", matched_plan),
                    ("counterfactual_ot", counterfactual_ot),
                    ("ctc_prior", ctc_prior),
                    ("ctc_train", ctc_train),
                ):
                    add_metrics(row, prefix, alignment_geometry(alignment))
                for prefix, left, right in (
                    ("direct_plan_vs_ot", matched_plan, counterfactual_ot),
                    ("plan_vs_ctc_prior", matched_plan, ctc_prior),
                    ("plan_vs_ctc_train", matched_plan, ctc_train),
                ):
                    add_metrics(row, prefix, plan_agreement(left, right))
                row["direct_plan_minus_ot_diag_mean_abs_dev"] = (
                    row["matched_plan__diag_mean_abs_dev"]
                    - row["counterfactual_ot__diag_mean_abs_dev"]
                )

                if spec_index == 0:
                    reference[cut.id] = {
                        "plan": matched_plan,
                        "ctc_prior": ctc_prior,
                        "ctc_train": ctc_train,
                    }
                    plan_cross = {"barycenter_mad": 0.0, "support_iou": 1.0, "total_variation": 0.0}
                    ctc_prior_cross = dict(plan_cross)
                    ctc_train_cross = dict(plan_cross)
                else:
                    plan_cross = plan_agreement(
                        matched_plan, reference[cut.id]["plan"]
                    )
                    ctc_prior_cross = plan_agreement(
                        ctc_prior, reference[cut.id]["ctc_prior"]
                    )
                    ctc_train_cross = plan_agreement(
                        ctc_train, reference[cut.id]["ctc_train"]
                    )
                add_metrics(row, "plan_vs_gw0", plan_cross)
                add_metrics(row, "ctc_prior_vs_gw0", ctc_prior_cross)
                add_metrics(row, "ctc_train_vs_gw0", ctc_train_cross)
                rows.append(row)

        if (batch_index + 1) % 10 == 0:
            print(f"{split}: completed {batch_index + 1} batches", flush=True)
    return rows


def bootstrap_summary(
    rows: Sequence[Dict[str, Any]],
    specs: Sequence[CheckpointSpec],
    splits: Sequence[str],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    excluded = {"split", "cut_id", "model", "lambda_gw", "num_frames", "num_tokens"}
    metric_keys = [key for key in rows[0] if key not in excluded]
    rng = np.random.default_rng(seed)
    summary: Dict[str, Any] = {"splits": {}}
    for split in splits:
        summary["splits"][split] = {}
        for spec in specs:
            selected = [
                row for row in rows if row["split"] == split and row["model"] == spec.name
            ]
            indices = rng.integers(
                0, len(selected), size=(samples, len(selected)), dtype=np.int32
            )
            metrics = {}
            for key in metric_keys:
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                boot = values[indices].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
                metrics[key] = {
                    "mean": float(values.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                }
            summary["splits"][split][spec.name] = {
                "lambda_gw": spec.lambda_gw,
                "num_utterances": len(selected),
                "metrics": metrics,
            }
    return summary


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = ["# TIMIT FGW plan-to-CTC transfer", ""]
    for split, models in summary["splits"].items():
        lines += [f"## {split}", "", "| Model | lambda | Direct bary | Direct support IoU | Direct diag delta | Plan-vs-0 bary | CTC-prior-vs-0 bary | CTC-train-vs-0 bary |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model, info in models.items():
            metric = info["metrics"]
            lines.append(
                f"| {model} | {info['lambda_gw']:.2f} | "
                f"{metric['direct_plan_vs_ot__barycenter_mad']['mean']:.5f} | "
                f"{metric['direct_plan_vs_ot__support_iou']['mean']:.4f} | "
                f"{metric['direct_plan_minus_ot_diag_mean_abs_dev']['mean']:+.5f} | "
                f"{metric['plan_vs_gw0__barycenter_mad']['mean']:.5f} | "
                f"{metric['ctc_prior_vs_gw0__barycenter_mad']['mean']:.5f} | "
                f"{metric['ctc_train_vs_gw0__barycenter_mad']['mean']:.5f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(args.output_dir / "log-evaluate-timit-fgw-transfer"))
    specs = parse_specs(args.checkpoint)
    device = resolve_device(args.device)
    _, graph = build_phone_graph(args.lang_dir, device=device)
    models = {}
    params = {}
    for spec in specs:
        models[spec.name], params[spec.name] = build_vi_model(
            spec.checkpoint, args.lang_dir, device=device
        )

    rows = []
    for split in args.splits:
        rows.extend(
            collect_split(
                split,
                get_split_dataloader(args, split),
                specs=specs,
                models=models,
                params=params,
                graph=graph,
                device=device,
            )
        )
    summary = bootstrap_summary(
        rows,
        specs=specs,
        splits=args.splits,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    with (args.output_dir / "utterance_metrics.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown(summary, args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
