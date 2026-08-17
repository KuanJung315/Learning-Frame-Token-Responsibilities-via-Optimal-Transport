from typing import Optional

import torch
from torch import Tensor


def encoder_lens_from_mask(
    memory_key_padding_mask: Optional[Tensor],
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> Tensor:
    if memory_key_padding_mask is None:
        return torch.full(
            (batch_size,),
            max_len,
            dtype=torch.long,
            device=device,
        )

    return (~memory_key_padding_mask).sum(dim=1).long()


def build_gated_log_probs_v2(
    log_p_nonblank: Tensor,
    alpha: Tensor,
    eps: float = 1.0e-5,
) -> Tensor:
    """
    Build exactly normalized VarCTC v2 emissions from a V-1 non-blank head.

    col 0:      log(1 - alpha_t)
    col 1..V-1: log(alpha_t) + log_p_nonblank[k]

    The non-blank head is already normalized across V-1 classes, so no
    renormalization against a learned blank column is needed.
    """
    alpha = alpha.float().clamp(eps, 1.0 - eps)
    log_blank = torch.log1p(-alpha)
    log_nonblank = torch.log(alpha).unsqueeze(-1) + log_p_nonblank.float()
    return torch.cat([log_blank.unsqueeze(-1), log_nonblank], dim=-1)


def mean_valid_frame_std(values: Tensor, lengths: Tensor) -> Tensor:
    stats = []
    for i, length in enumerate(lengths.detach().cpu().tolist()):
        if length > 1:
            stats.append(values[i, :length].float().std(unbiased=False))
    if not stats:
        return values.new_tensor(0.0)
    return torch.stack(stats).mean()


def mean_valid_abs_diff(x: Tensor, y: Tensor, lengths: Tensor) -> Tensor:
    vals = []
    for i, length in enumerate(lengths.detach().cpu().tolist()):
        if length > 0:
            vals.append((x[i, :length].float() - y[i, :length].float()).abs().mean())
    if not vals:
        return x.new_tensor(0.0)
    return torch.stack(vals).mean()
