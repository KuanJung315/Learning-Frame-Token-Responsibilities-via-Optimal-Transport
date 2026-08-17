#!/usr/bin/env python3
"""Run evaluate_vi_alignment_metrics_aishell.py with AdaMER in the *baseline* slot.

AdaMER is plain CTC at inference (beta has no effect), so its posterior geometry
must be measured on plain CTC log-probs -- i.e. the baseline path, NOT the VI
gated candidate path. The only wrinkle is that AdaMER checkpoints carry an extra
adamer_raw_beta parameter, so we swap in AdaMERConformer *only* while the baseline
model is built/loaded; the VI candidate still loads with the plain Conformer.
This yields AdaMER's geometry metrics (baseline_*) alongside the VI model's
(candidate_*), on the same deterministic first-N dev cuts as the paper table.
"""

import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_ASR_DIR = _RECIPE_DIR.parent
for _p in (_RECIPE_DIR, _ASR_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate_vi_alignment_metrics_aishell as ev  # noqa: E402
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
