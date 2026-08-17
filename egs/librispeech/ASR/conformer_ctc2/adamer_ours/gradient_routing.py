"""Gradient-routing helpers for AdaMER + Ours compatibility experiments."""

from torch import Tensor


def with_gradient_scale(value: Tensor, scale: float) -> Tensor:
    """Preserve the forward value while scaling its backward gradient."""
    return value.detach() + float(scale) * (value - value.detach())
