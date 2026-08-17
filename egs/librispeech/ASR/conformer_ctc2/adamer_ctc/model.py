"""Conformer wrapper carrying the learned AdaMER dual variable."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from conformer import Conformer as BaseConformer


_INITIAL_BETA = 0.2


def set_initial_beta(value: float) -> None:
    global _INITIAL_BETA
    if value <= 0:
        raise ValueError("AdaMER initial beta must be positive")
    _INITIAL_BETA = value


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class AdaMERConformer(BaseConformer):
    """The baseline Conformer plus a learnable non-negative AdaMER weight."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adamer_raw_beta = nn.Parameter(
            torch.tensor(_inverse_softplus(_INITIAL_BETA))
        )

    def adamer_beta(self) -> Tensor:
        # Softplus enforces the beta >= 0 constraint from the dual problem.
        return nn.functional.softplus(self.adamer_raw_beta)

    def forward(self, *args, **kwargs):
        nnet_output, encoder_memory, memory_mask = super().forward(*args, **kwargs)
        # Keep beta in the DDP forward graph. Its real gradient still comes
        # only from L_beta because this dependency has exactly zero derivative.
        nnet_output = nnet_output + self.adamer_raw_beta * 0.0
        return nnet_output, encoder_memory, memory_mask
