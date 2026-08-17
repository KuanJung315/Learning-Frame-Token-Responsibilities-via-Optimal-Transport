#!/usr/bin/env python3
"""Unit tests for label-prior CTC helpers."""

import sys
from pathlib import Path

import torch

_RECIPE_DIR = Path(__file__).resolve().parent.parent
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

from label_prior_ctc.model import apply_label_prior


def test_apply_label_prior_matches_log_space_equation():
    log_probs = torch.log_softmax(
        torch.tensor([[[1.0, 0.5, -0.5], [0.1, 0.2, 0.3]]]),
        dim=-1,
    )
    label_prior = torch.tensor([0.5, 0.25, 0.25])

    adjusted = apply_label_prior(
        log_probs=log_probs,
        label_prior=label_prior,
        alpha=0.7,
        floor=1.0e-8,
        enabled=True,
    )

    expected = log_probs - 0.7 * label_prior.log().view(1, 1, -1)
    assert torch.allclose(adjusted, expected)


def test_alpha_zero_returns_input_tensor():
    log_probs = torch.randn(2, 3, 4)
    label_prior = torch.full((4,), 0.25)

    adjusted = apply_label_prior(
        log_probs=log_probs,
        label_prior=label_prior,
        alpha=0.0,
        floor=1.0e-8,
        enabled=True,
    )

    assert adjusted is log_probs


def test_disabled_label_prior_returns_input_tensor():
    log_probs = torch.randn(2, 3, 4)
    label_prior = torch.tensor([0.7, 0.1, 0.1, 0.1])

    adjusted = apply_label_prior(
        log_probs=log_probs,
        label_prior=label_prior,
        alpha=0.3,
        floor=1.0e-8,
        enabled=False,
    )

    assert adjusted is log_probs
