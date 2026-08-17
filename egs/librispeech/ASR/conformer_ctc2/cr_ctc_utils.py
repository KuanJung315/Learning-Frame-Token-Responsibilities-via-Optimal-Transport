"""Shared augmentation and consistency utilities for CR-CTC recipes."""

from typing import Tuple

import torch
from torch import Tensor


def with_gradient_scale(value: Tensor, scale: float) -> Tensor:
    """Preserve the forward value while scaling its backward gradient."""
    return value.detach() + float(scale) * (value - value.detach())


def linear_warmup_weight(
    batch_idx: int,
    start: int,
    steps: int,
    target: float,
) -> float:
    """Linearly ramp from zero to target after the requested start batch."""
    if batch_idx < start:
        return 0.0
    if steps <= 0:
        return float(target)
    progress = min(max((batch_idx - start) / steps, 0.0), 1.0)
    return float(target) * progress


def mask_span(
    max_len: int,
    max_width: int,
    device: torch.device,
) -> Tuple[int, int]:
    width_cap = min(max_len, max_width)
    if width_cap <= 0:
        return 0, 0
    width = int(torch.randint(0, width_cap + 1, (1,), device=device).item())
    if width <= 0:
        return 0, 0
    start = int(torch.randint(0, max_len - width + 1, (1,), device=device).item())
    return start, width


def make_cr_view(feature: Tensor, feature_lens: Tensor, params) -> Tensor:
    """Create the matched second view with additional time/frequency masks."""
    if params.cr_num_time_masks <= 0 and params.cr_num_feature_masks <= 0:
        return feature

    out = feature.clone()
    batch_size, _, num_features = out.shape
    device = out.device

    for b in range(batch_size):
        valid_t = int(feature_lens[b].item())
        for _ in range(params.cr_num_time_masks):
            start, width = mask_span(
                valid_t, int(params.cr_time_mask_size), device=device
            )
            if width > 0:
                out[b, start : start + width, :] = float(params.cr_mask_value)

        for _ in range(params.cr_num_feature_masks):
            start, width = mask_span(
                num_features, int(params.cr_feature_mask_size), device=device
            )
            if width > 0:
                out[b, :valid_t, start : start + width] = float(
                    params.cr_mask_value
                )

    return out


def cr_ctc_consistency_loss(
    log_probs_a: Tensor,
    log_probs_b: Tensor,
    lengths_a: Tensor,
    lengths_b: Tensor,
    stop_gradient: bool,
    temperature: float,
) -> Tensor:
    """Return per-utterance symmetric KL over valid CTC frames."""
    t_max = min(log_probs_a.size(1), log_probs_b.size(1))
    log_probs_a = log_probs_a[:, :t_max]
    log_probs_b = log_probs_b[:, :t_max]
    lengths = torch.minimum(lengths_a, lengths_b).clamp(max=t_max)

    temp = max(float(temperature), 1.0e-6)
    log_pa = torch.log_softmax(log_probs_a.float() / temp, dim=-1)
    log_pb = torch.log_softmax(log_probs_b.float() / temp, dim=-1)
    pa = log_pa.exp()
    pb = log_pb.exp()

    if stop_gradient:
        kl_ab = (pa.detach() * (log_pa.detach() - log_pb)).sum(dim=-1)
        kl_ba = (pb.detach() * (log_pb.detach() - log_pa)).sum(dim=-1)
    else:
        kl_ab = (pa * (log_pa - log_pb)).sum(dim=-1)
        kl_ba = (pb * (log_pb - log_pa)).sum(dim=-1)

    frame_kl = 0.5 * (kl_ab + kl_ba) * (temp * temp)
    frame_kl = torch.nan_to_num(frame_kl, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (
        torch.arange(t_max, device=log_probs_a.device)[None, :]
        < lengths[:, None]
    )
    return (frame_kl * valid.to(frame_kl.dtype)).sum(dim=1)
