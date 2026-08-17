#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from evaluate_timit_alignment import (
    boundary_match_counts,
    gold_phone_alignment,
    plan_alignment,
)
from ot_prior_v2 import vi_ot_loss_v2
from timit_eval_common import (
    add_data_arguments,
    build_phone_graph,
    build_vi_model,
    get_split_dataloader,
    resolve_device,
    vi_batch_outputs,
)

from icefall.utils import setup_logger


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--vi-checkpoint",
        type=Path,
        default=Path("conformer_ctc2/exp_vi_ot_phone_small/epoch-50.pt"),
    )
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_phone"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--max-cuts", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--config-names",
        nargs="*",
        default=None,
        help="Optional subset of diagnostic config names to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/timit_ot_plan_diagnosis"),
    )
    add_data_arguments(parser)
    return parser


def configs(params) -> List[Dict[str, Any]]:
    current = {
        "gate": "posterior",
        "column_marginal_type": getattr(params, "col_marginal_type", "acoustic"),
        "alpha_smooth_mix": float(getattr(params, "alpha_smooth_mix", 0.1)),
        "token_prior_sigma": float(getattr(params, "ot_token_prior_sigma", 0.15)),
        "token_prior_score_temp": float(
            getattr(params, "ot_token_prior_score_temp", 1.0)
        ),
        "token_prior_floor": float(getattr(params, "ot_token_prior_floor", 0.05)),
        "eps": float(getattr(params, "ot_eps", 0.3)),
        "iters": int(getattr(params, "ot_iters", 10)),
        "beta_pos": float(getattr(params, "ot_beta_pos", 1.0)),
        "monotonic_centers": False,
    }

    def make(name: str, **updates) -> Dict[str, Any]:
        config = {"name": name, **current}
        config.update(updates)
        return config

    return [
        make("current_posterior"),
        make("current_prior", gate="prior"),
        make("beta5_eps03", beta_pos=5.0, iters=30),
        make("beta10_eps03", beta_pos=10.0, iters=30),
        make("beta10_eps01", beta_pos=10.0, eps=0.1, iters=40),
        make("beta20_eps01", beta_pos=20.0, eps=0.1, iters=50),
        make("beta50_eps005", beta_pos=50.0, eps=0.05, iters=80),
        make(
            "beta50_eps005_monotonic",
            beta_pos=50.0,
            eps=0.05,
            iters=80,
            monotonic_centers=True,
        ),
        make("beta75_eps005", beta_pos=75.0, eps=0.05, iters=100),
        make("beta100_eps005", beta_pos=100.0, eps=0.05, iters=120),
        make("beta100_eps002", beta_pos=100.0, eps=0.02, iters=160),
        make("beta200_eps002", beta_pos=200.0, eps=0.02, iters=200),
        make(
            "prior_beta50_eps005",
            gate="prior",
            beta_pos=50.0,
            eps=0.05,
            iters=80,
        ),
        make(
            "uniform_beta10_eps01",
            column_marginal_type="uniform",
            beta_pos=10.0,
            eps=0.1,
            iters=40,
        ),
        make(
            "uniform_beta20_eps01",
            column_marginal_type="uniform",
            beta_pos=20.0,
            eps=0.1,
            iters=50,
        ),
        make(
            "uniform_beta20_eps01_monotonic",
            column_marginal_type="uniform",
            beta_pos=20.0,
            eps=0.1,
            iters=50,
            monotonic_centers=True,
        ),
        make(
            "acoustic_nofloor_beta10_eps01",
            token_prior_floor=0.0,
            beta_pos=10.0,
            eps=0.1,
            iters=40,
        ),
        make(
            "acoustic_sigma005_beta10_eps01",
            token_prior_sigma=0.05,
            beta_pos=10.0,
            eps=0.1,
            iters=40,
        ),
        make(
            "acoustic_sigma030_beta10_eps01",
            token_prior_sigma=0.30,
            beta_pos=10.0,
            eps=0.1,
            iters=40,
        ),
        make(
            "smooth0_beta10_eps01",
            alpha_smooth_mix=0.0,
            beta_pos=10.0,
            eps=0.1,
            iters=40,
        ),
        make(
            "prior_beta10_eps01",
            gate="prior",
            beta_pos=10.0,
            eps=0.1,
            iters=40,
        ),
    ]


def compute_plan(output, labels: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    alpha = output.alpha_post if config["gate"] == "posterior" else output.alpha_prior
    _, plan = vi_ot_loss_v2(
        log_p_nonblank=output.log_p_nonblank,
        alpha=alpha,
        labels=labels,
        column_marginal_type=config["column_marginal_type"],
        alpha_smooth_mix=config["alpha_smooth_mix"],
        token_prior_sigma=config["token_prior_sigma"],
        token_prior_score_temp=config["token_prior_score_temp"],
        token_prior_floor=config["token_prior_floor"],
        eps=config["eps"],
        iters=config["iters"],
        beta_pos=config["beta_pos"],
        return_plan=True,
    )
    if plan is None:
        raise ValueError("Empty OT plan")
    return plan.detach().cpu()


def main() -> None:
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(args.output_dir / "log-diagnose-timit-ot-plan"))
    device = resolve_device(args.device)
    _, graph = build_phone_graph(args.lang_dir, device=device)
    vi_model, vi_params = build_vi_model(
        args.vi_checkpoint, args.lang_dir, device=device
    )
    sweep = configs(vi_params)
    if args.config_names:
        requested = set(args.config_names)
        sweep = [config for config in sweep if config["name"] in requested]
        found = {config["name"] for config in sweep}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Unknown diagnostic configs: {missing}")
    accum: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "center_errors": [],
            "boundary_abs_errors": [],
            "backward_count": 0,
            "transition_count": 0,
            "matches_20ms": 0,
            "matches_50ms": 0,
            "boundary_count": 0,
            "utterances": 0,
        }
    )

    for batch_idx, batch in enumerate(get_split_dataloader(args, args.split)):
        vi_outputs, supervision_ids = vi_batch_outputs(
            vi_model,
            batch,
            graph=graph,
            device=device,
            gate="posterior",
        )
        sequence_idx = batch["supervisions"]["sequence_idx"].tolist()
        for sup_idx, cut in enumerate(batch["supervisions"]["cut"]):
            output = vi_outputs[int(sequence_idx[sup_idx])]
            labels = torch.tensor(supervision_ids[sup_idx], dtype=torch.long)
            gold = gold_phone_alignment(cut)
            for config in sweep:
                plan = compute_plan(output, labels=labels, config=config)
                pred = plan_alignment(
                    plan,
                    frame_shift=float(getattr(cut.features, "frame_shift", 0.01)),
                    subsampling_factor=int(
                        getattr(vi_params, "subsampling_factor", 4)
                    ),
                    enforce_monotonic_centers=config["monotonic_centers"],
                )
                values = accum[config["name"]]
                values["center_errors"].extend(
                    abs(gold_value - pred_value)
                    for gold_value, pred_value in zip(
                        gold["centers_sec"], pred["centers_sec"]
                    )
                )
                values["boundary_abs_errors"].extend(
                    abs(gold_value - pred_value)
                    for gold_value, pred_value in zip(
                        gold["boundaries_sec"], pred["boundaries_sec"]
                    )
                )
                diffs = np.diff(pred["centers_sec"])
                values["backward_count"] += int((diffs < 0).sum())
                values["transition_count"] += len(diffs)
                for tolerance_ms in (20, 50):
                    matches, _, gold_count = boundary_match_counts(
                        pred["boundaries_sec"],
                        gold["boundaries_sec"],
                        tolerance_sec=tolerance_ms / 1000.0,
                    )
                    values[f"matches_{tolerance_ms}ms"] += matches
                    if tolerance_ms == 20:
                        values["boundary_count"] += gold_count
                values["utterances"] += 1
        if batch_idx % 10 == 0:
            logging.info("processed batch %s", batch_idx)

    summary = {}
    for config in sweep:
        values = accum[config["name"]]
        boundary_count = max(values["boundary_count"], 1)
        summary[config["name"]] = {
            "config": config,
            "utterances": values["utterances"],
            "center_mae_ms": 1000.0 * float(np.mean(values["center_errors"])),
            "boundary_mae_ms": 1000.0 * float(
                np.mean(values["boundary_abs_errors"])
            ),
            "boundary_f1_20ms": values["matches_20ms"] / boundary_count,
            "boundary_f1_50ms": values["matches_50ms"] / boundary_count,
            "center_backward_rate": values["backward_count"]
            / max(values["transition_count"], 1),
        }

    ranked = sorted(summary.items(), key=lambda item: item[1]["center_mae_ms"])
    for name, values in ranked:
        logging.info(
            "%-32s center=%8.2fms boundary=%8.2fms F1@20=%6.4f "
            "F1@50=%6.4f backward=%6.4f",
            name,
            values["center_mae_ms"],
            values["boundary_mae_ms"],
            values["boundary_f1_20ms"],
            values["boundary_f1_50ms"],
            values["center_backward_rate"],
        )
    with open(args.output_dir / f"{args.split}-ot-plan-sweep.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
