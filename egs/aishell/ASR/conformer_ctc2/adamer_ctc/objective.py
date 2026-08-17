"""AdaMER-CTC objectives with the paper's stop-gradient separation."""

from typing import Tuple

import torch
from torch import Tensor


def adamer_objectives(
    ctc_loss: Tensor,
    path_entropy: Tensor,
    target_entropy: Tensor,
    beta: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return EnCTC loss, beta loss, and the entropy regularizer.

    Gradients from the EnCTC objective flow through path entropy but not beta.
    Gradients from the beta objective flow through beta but not path entropy.
    """
    entropy_regularizer = -beta.detach() * path_entropy.sum()
    enctc_loss = ctc_loss + entropy_regularizer
    beta_loss = beta * (path_entropy.detach() - target_entropy).sum()
    return enctc_loss, beta_loss, entropy_regularizer
