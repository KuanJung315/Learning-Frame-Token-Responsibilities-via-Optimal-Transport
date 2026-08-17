#!/usr/bin/env python3
"""Decode label-prior CTC checkpoints with the matching Conformer wrapper.

By default this reproduces plain CTC decoding. With ``--label-prior-alpha > 0``
the CTC emissions are adjusted to ``log y - alpha * log P(k)`` at inference,
using the prior saved in the checkpoint. This is train/inference-consistent and
matches Huang et al., which applies label priors during Viterbi search as well
as training. ``alpha`` should normally equal the training value; the baseline
(alpha=0) decode is unchanged, so the protocol stays symmetric across systems.
"""

import importlib.util
import math
import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_ASR_DIR = _RECIPE_DIR.parent
for _path in (_RECIPE_DIR, _ASR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from label_prior_ctc.model import LabelPriorConformer  # noqa: E402


def _load_baseline_decode():
    decode_path = _RECIPE_DIR / "decode.py"
    spec = importlib.util.spec_from_file_location(
        "conformer_ctc2_baseline_decode",
        decode_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load baseline decoder from {decode_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    baseline_decode = _load_baseline_decode()
    baseline_decode.Conformer = LabelPriorConformer

    # Add label-prior decode options to the baseline parser. main() looks up
    # get_parser via module globals, so patching the attribute is enough.
    _orig_get_parser = baseline_decode.get_parser

    def get_parser():
        parser = _orig_get_parser()
        parser.add_argument(
            "--label-prior-alpha",
            type=float,
            default=0.0,
            help=(
                "Apply log y - alpha * log P(k) at decode using the prior in "
                "the checkpoint. 0 (default) is plain CTC decoding. Set equal "
                "to the training alpha for train/inference-consistent decoding."
            ),
        )
        parser.add_argument(
            "--label-prior-floor",
            type=float,
            default=math.exp(-12.0),
            help="Numerical floor for the decode-time prior. Defaults to exp(-12).",
        )
        return parser

    baseline_decode.get_parser = get_parser

    # Push the parsed options onto the model instance for each batch.
    _orig_decode_one_batch = baseline_decode.decode_one_batch

    def decode_one_batch(params, model, *args, **kwargs):
        alpha = float(getattr(params, "label_prior_alpha", 0.0))
        model.decode_alpha = alpha
        model.decode_floor = float(getattr(params, "label_prior_floor", math.exp(-12.0)))
        model.apply_prior_in_forward = alpha > 0.0
        return _orig_decode_one_batch(params, model, *args, **kwargs)

    baseline_decode.decode_one_batch = decode_one_batch

    baseline_decode.main()


if __name__ == "__main__":
    main()
