#!/usr/bin/env python3
"""Decode AdaMER+Ours checkpoints with the VI blank-prior decoder."""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_ASR_DIR = _RECIPE_DIR.parent
for _path in (_RECIPE_DIR, _ASR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import decode_vi_blank_v2 as baseline_decode  # noqa: E402


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class AdaMEROursConformerVIV2ForDecode(baseline_decode.ConformerVIV2ForDecode):
    """Decode wrapper carrying the extra AdaMER beta checkpoint parameter."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adamer_raw_beta = nn.Parameter(torch.tensor(_inverse_softplus(0.2)))

    def adamer_beta(self):
        return nn.functional.softplus(self.adamer_raw_beta)


def main() -> None:
    baseline_decode.ConformerVIV2ForDecode = AdaMEROursConformerVIV2ForDecode
    baseline_decode.main()


if __name__ == "__main__":
    main()
