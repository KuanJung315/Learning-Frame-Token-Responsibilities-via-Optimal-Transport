#!/usr/bin/env python3
"""Decode AdaMER checkpoints with the standard-CTC logit-bias sweep tool.

Wraps conformer_ctc2/decode_ctc_bias.py, swapping in AdaMERConformer so the
averaged AdaMER checkpoint (which carries the extra beta parameter) loads under
strict state-dict matching. The beta parameter does not affect inference.
"""

import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_ASR_DIR = _RECIPE_DIR.parent
for _path in (_RECIPE_DIR, _ASR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import decode_ctc_bias as baseline_decode_ctc_bias  # noqa: E402
from adamer_ctc.model import AdaMERConformer  # noqa: E402


def main() -> None:
    baseline_decode_ctc_bias.Conformer = AdaMERConformer
    baseline_decode_ctc_bias.main()


if __name__ == "__main__":
    main()
