#!/usr/bin/env python3
"""Decode AISHELL checkpoints produced by adamer_ctc/train.py."""

import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_ASR_DIR = _RECIPE_DIR.parent
for _path in (_RECIPE_DIR, _ASR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import decode as baseline_decode  # noqa: E402
from adamer_ctc.model import AdaMERConformer  # noqa: E402


def main() -> None:
    # The beta parameter does not affect inference, but keeping it in the model
    # lets strict checkpoint averaging load AdaMER checkpoints.
    baseline_decode.Conformer = AdaMERConformer
    baseline_decode.main()


if __name__ == "__main__":
    main()
