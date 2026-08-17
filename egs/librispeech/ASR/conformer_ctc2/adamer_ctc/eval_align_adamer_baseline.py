#!/usr/bin/env python3
"""Run evaluate_vi_alignment_metrics.py with AdaMER in the *baseline* slot.

The baseline slot builds a plain Conformer and strict-loads the checkpoint;
AdaMER checkpoints carry an extra adamer_raw_beta parameter, so we swap in
AdaMERConformer *only* while the baseline model is constructed/loaded (the VI
candidate must still load with the plain Conformer). This yields AdaMER's
alignment metrics (baseline_*) alongside VFTA's (candidate_*).
"""

import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_ASR_DIR = _RECIPE_DIR.parent
for _p in (_RECIPE_DIR, _ASR_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate_vi_alignment_metrics as ev  # noqa: E402
from adamer_ctc.model import AdaMERConformer  # noqa: E402

_orig_build_baseline = ev._build_baseline_model


def _build_baseline_with_adamer(*args, **kwargs):
    saved = ev.Conformer
    ev.Conformer = AdaMERConformer  # AdaMER checkpoint has adamer_raw_beta
    try:
        return _orig_build_baseline(*args, **kwargs)
    finally:
        ev.Conformer = saved  # restore so the VI candidate loads normally


ev._build_baseline_model = _build_baseline_with_adamer

if __name__ == "__main__":
    ev.main()
