#!/usr/bin/env python3
"""
Bake a VI decode-time prior logit bias into blank_prior.linear.bias.

For use-averaged-model decoding, icefall computes the averaged model from the
`model_avg` fields of two endpoint checkpoints. To make bake-in exactly match
decode-time --prior-logit-bias, this script shifts both `model` and `model_avg`
in the start/end checkpoints used by decoding.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import torch


BIAS_KEYS = (
    "blank_prior.linear.bias",
    "module.blank_prior.linear.bias",
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-exp", type=Path, required=True)
    parser.add_argument("--dst-exp", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--bias-shift", type=float, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def shift_state_dict(
    state_dict: Dict[str, torch.Tensor],
    bias_shift: float,
    state_name: str,
) -> Dict[str, float]:
    for key in BIAS_KEYS:
        if key in state_dict:
            before = state_dict[key].detach().float().mean().item()
            state_dict[key] = state_dict[key] + bias_shift
            after = state_dict[key].detach().float().mean().item()
            return {
                "state": state_name,
                "key": key,
                "before_mean": before,
                "after_mean": after,
            }

    raise KeyError(
        f"Could not find blank prior bias in {state_name}. "
        f"Tried keys: {', '.join(BIAS_KEYS)}"
    )


def bake_checkpoint(
    src: Path,
    dst: Path,
    bias_shift: float,
    force: bool,
) -> Iterable[Dict[str, float]]:
    if dst.exists() and not force:
        return []

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    records = []
    for state_name in ("model", "model_avg"):
        if state_name in checkpoint:
            records.append(
                shift_state_dict(
                    checkpoint[state_name],
                    bias_shift=bias_shift,
                    state_name=state_name,
                )
            )

    if not records:
        raise KeyError(f"No model/model_avg state found in {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dst)
    return records


def main() -> None:
    args = get_parser().parse_args()
    args.dst_exp.mkdir(parents=True, exist_ok=True)

    manifest = {
        "src_exp": str(args.src_exp),
        "dst_exp": str(args.dst_exp),
        "epochs": args.epochs,
        "bias_shift": args.bias_shift,
        "records": {},
    }

    for epoch in args.epochs:
        src = args.src_exp / f"epoch-{epoch}.pt"
        dst = args.dst_exp / f"epoch-{epoch}.pt"
        if not src.is_file():
            raise FileNotFoundError(src)

        records = list(
            bake_checkpoint(
                src=src,
                dst=dst,
                bias_shift=args.bias_shift,
                force=args.force,
            )
        )
        manifest["records"][f"epoch-{epoch}"] = records
        if records:
            print(f"Baked {src} -> {dst}")
            for record in records:
                print(
                    f"  {record['state']}:{record['key']} "
                    f"{record['before_mean']:.6g} -> {record['after_mean']:.6g}"
                )
        else:
            print(f"Skip existing {dst}; pass --force to overwrite")

    manifest_path = args.dst_exp / "bake_prior_bias_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
