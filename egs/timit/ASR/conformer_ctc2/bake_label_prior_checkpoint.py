#!/usr/bin/env python3
"""Fold a label-prior penalty into the TIMIT CTC output-layer bias."""

import argparse
import math
from pathlib import Path

import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--floor", type=float, default=None)
    return parser


def bake(checkpoint: dict, alpha: float = None, floor: float = None) -> dict:
    state = checkpoint["model"]
    prior = state["label_prior"].float()
    if not bool(state["label_prior_ready"].item()):
        raise ValueError("Label prior is not ready")
    alpha = float(
        checkpoint.get("label_prior_alpha", 0.3) if alpha is None else alpha
    )
    floor = float(
        checkpoint.get("label_prior_floor", math.exp(-12.0))
        if floor is None
        else floor
    )
    prior = prior.clamp_min(floor)
    prior = prior / prior.sum()
    penalty = alpha * prior.log()
    bias_key = "encoder_output_layer.1.bias"
    if bias_key not in state:
        raise KeyError(f"Missing {bias_key}")
    state[bias_key] = state[bias_key] - penalty.to(state[bias_key])
    del state["label_prior"]
    del state["label_prior_ready"]
    checkpoint["baked_label_prior_alpha"] = alpha
    checkpoint["baked_label_prior_floor"] = floor
    checkpoint["baked_label_prior"] = prior
    checkpoint["model_type"] = "baseline_label_prior_baked"
    return checkpoint


def main() -> None:
    args = get_parser().parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    checkpoint = bake(checkpoint, alpha=args.alpha, floor=args.floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved baked label-prior checkpoint to {args.output}")


if __name__ == "__main__":
    main()
