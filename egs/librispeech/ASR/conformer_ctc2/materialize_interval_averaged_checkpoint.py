#!/usr/bin/env python3
"""Materialize an evaluation-only checkpoint for an interval-averaged model.

Icefall stores a running ``model_avg`` from the beginning of training.  Given
two epoch checkpoints, the standard averaged-model identity recovers the mean
over updates in ``(start, end]`` without naively averaging endpoint weights.
The output intentionally omits optimizer/scheduler states and is suitable for
the forced-alignment evaluators that load the ``model`` key.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Dict

import torch


_HEAVY_KEYS = {
    "model",
    "model_avg",
    "optimizer",
    "scheduler",
    "grad_scaler",
    "sampler",
}


def _average_state_dict(
    state_dict_1: Dict[str, torch.Tensor],
    state_dict_2: Dict[str, torch.Tensor],
    weight_1: float,
    weight_2: float,
    scaling_factor: float,
) -> None:
    """Minimal dependency-free form of icefall's in-place state averaging."""
    unique: Dict[int, str] = {}
    for key, value in state_dict_1.items():
        unique.setdefault(value.data_ptr(), key)
    for key in unique.values():
        value = state_dict_1[key]
        if value.is_floating_point():
            value.mul_(weight_1)
            value.add_(state_dict_2[key].to(value.device), alpha=weight_2)
            value.mul_(scaling_factor)


def _load_for_average(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_avg" not in checkpoint:
        raise ValueError(f"{path} has no model_avg state")
    if "batch_idx_train" not in checkpoint or "average_period" not in checkpoint:
        raise ValueError(f"{path} lacks averaged-model counters")
    return checkpoint


def interval_averaged_state(
    start_checkpoint: Dict[str, Any],
    end_checkpoint: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Recover the update average in ``(start, end]`` in-place on end state."""
    average_period = int(start_checkpoint["average_period"])
    if average_period != int(end_checkpoint["average_period"]):
        raise ValueError("start/end average_period values differ")
    start_batch = (
        int(start_checkpoint["batch_idx_train"]) // average_period
    ) * average_period
    end_batch = (
        int(end_checkpoint["batch_idx_train"]) // average_period
    ) * average_period
    interval = end_batch - start_batch
    if interval <= 0:
        raise ValueError(
            f"end checkpoint must follow start: {start_batch} -> {end_batch}"
        )

    end_state = end_checkpoint["model_avg"]
    start_state = start_checkpoint["model_avg"]
    if end_state.keys() != start_state.keys():
        raise ValueError("start/end model_avg keys differ")
    weight_end = end_batch / interval
    weight_start = 1.0 - weight_end
    _average_state_dict(
        state_dict_1=end_state,
        state_dict_2=start_state,
        weight_1=1.0,
        weight_2=weight_start / weight_end,
        scaling_factor=weight_end,
    )
    return end_state


def materialize(start: Path, end: Path, output: Path) -> None:
    end_checkpoint = _load_for_average(end)
    metadata = {
        key: value for key, value in end_checkpoint.items() if key not in _HEAVY_KEYS
    }
    end_state = end_checkpoint["model_avg"]
    end_minimal = {
        "model_avg": end_state,
        "batch_idx_train": end_checkpoint["batch_idx_train"],
        "average_period": end_checkpoint["average_period"],
    }
    del end_checkpoint
    gc.collect()

    start_checkpoint = _load_for_average(start)
    start_minimal = {
        "model_avg": start_checkpoint["model_avg"],
        "batch_idx_train": start_checkpoint["batch_idx_train"],
        "average_period": start_checkpoint["average_period"],
    }
    del start_checkpoint
    gc.collect()

    state = interval_averaged_state(start_minimal, end_minimal)
    metadata.update(
        {
            "model": state,
            "evaluation_state": "interval_averaged_model",
            "evaluation_average_start": str(start),
            "evaluation_average_end": str(end),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metadata, output)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--end-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    if not args.start_checkpoint.is_file():
        raise FileNotFoundError(args.start_checkpoint)
    if not args.end_checkpoint.is_file():
        raise FileNotFoundError(args.end_checkpoint)
    materialize(args.start_checkpoint, args.end_checkpoint, args.output)


if __name__ == "__main__":
    main()
