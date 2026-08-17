#!/usr/bin/env python3
"""Prune redundant checkpoints from completed VFTA-FGW experiments.

The default is a dry run.  Only direct checkpoint children of ``exp_*``
directories are considered; logs, TensorBoard files, and evaluation outputs
are never touched.
"""

import argparse
import re
from pathlib import Path


EXPECTED_ROOT = Path(
    "/work/u4218021/icefall/egs/librispeech/ASR/conformer_ctc2/vfta_fgw"
)
ACTIVE_EXPERIMENTS = {
    "exp_li100h_phone_gw0p1_w10_seed43",
    "exp_li100h_phone_gw0p1_w10p03_seed43",
    "exp_li100h_phone_gw0p1_w10_seed44",
    "exp_li100h_phone_gw0p1_w10p03_seed44",
}
EPOCH_PATTERN = re.compile(r"epoch-(\d+)\.pt")
BATCH_PATTERN = re.compile(r"checkpoint-\d+\.pt")
BEST_CHECKPOINTS = {"best-train-loss.pt", "best-valid-loss.pt"}


def retained_epochs(directory: Path) -> set[int]:
    if "phone" in directory.name:
        keep = {20, 30}
        if "metric_corrected_psd" in directory.name:
            keep.update({10, 15})
        return keep
    if "li960h" in directory.name:
        return {20, 30}
    return {30, 40}


def removable_checkpoints(directory: Path, keep: set[int]) -> list[Path]:
    targets = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        epoch_match = EPOCH_PATTERN.fullmatch(path.name)
        removable_epoch = (
            epoch_match is not None and int(epoch_match.group(1)) not in keep
        )
        if (
            removable_epoch
            or path.name in BEST_CHECKPOINTS
            or BATCH_PATTERN.fullmatch(path.name)
        ):
            targets.append(path)
    return sorted(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if root != EXPECTED_ROOT:
        raise RuntimeError(f"Refusing unexpected root: {root}")

    total_bytes = 0
    total_files = 0
    total_dirs = 0
    for directory in sorted(root.glob("exp_*")):
        if not directory.is_dir() or directory.name in ACTIVE_EXPERIMENTS:
            continue
        keep = retained_epochs(directory)
        targets = removable_checkpoints(directory, keep)
        if not targets:
            continue

        existing_anchors = {
            int(match.group(1))
            for path in directory.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and (match := EPOCH_PATTERN.fullmatch(path.name)) is not None
            and int(match.group(1)) in keep
        }
        if existing_anchors != keep:
            raise RuntimeError(
                f"Refusing {directory.name}: expected anchors {sorted(keep)}, "
                f"found {sorted(existing_anchors)}"
            )

        size = sum(path.stat().st_size for path in targets)
        print(
            f"{'DELETE' if args.execute else 'WOULD_DELETE'} "
            f"{size / 2**30:8.2f} GiB {len(targets):3d} files "
            f"keep={sorted(keep)} {directory.name}"
        )
        if args.execute:
            for path in targets:
                path.unlink()
        total_bytes += size
        total_files += len(targets)
        total_dirs += 1

    print(
        f"TOTAL {'DELETED' if args.execute else 'WOULD_DELETE'} "
        f"{total_bytes / 2**40:.3f} TiB ({total_bytes / 2**30:.1f} GiB), "
        f"{total_files} files across {total_dirs} directories"
    )


if __name__ == "__main__":
    main()
