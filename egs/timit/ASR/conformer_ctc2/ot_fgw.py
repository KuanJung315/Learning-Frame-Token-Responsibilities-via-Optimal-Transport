from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ot_prior_v2 import (
    _NEG,
    _frac_positions,
    _make_alpha_row_marginal,
    _make_bpe_length_column_marginal,
    _make_soft_column_marginal,
    _make_uniform_column_marginal,
    _masked_normalize,
    _mix_uniform_batched,
    _sinkhorn,
    _sinkhorn_batched,
    vi_ot_loss_v2,
    vi_ot_loss_v2_batched,
)


@torch.no_grad()
def _make_pos_cost(
    T: int, U: int, device: torch.device, beta_pos: float
) -> Optional[Tensor]:
    if beta_pos <= 0:
        return None
    t_pos = torch.linspace(0, 1, T, device=device).unsqueeze(1)
    u_pos = torch.linspace(0, 1, U, device=device).unsqueeze(0)
    return beta_pos * (t_pos - u_pos).pow(2)


@torch.no_grad()
def _compute_dx(acoustic_features: Tensor) -> Tensor:
    """Pairwise cosine distance over encoder-aligned raw Fbank features."""
    x = F.normalize(acoustic_features.float(), dim=-1)
    return (1.0 - x @ x.transpose(0, 1)).clamp_min(0.0)


@torch.no_grad()
def _compute_dy(U: int, device: torch.device) -> Tensor:
    """Pairwise normalized distance over transcript-token positions."""
    idx = torch.arange(U, device=device, dtype=torch.float32)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() / max(float(U), 1.0)


@torch.no_grad()
def _gw_gradient(D_X: Tensor, D_Y: Tensor, A: Tensor) -> Tensor:
    """Gradient of the squared-loss GW energy with respect to A."""
    row_mass = A.sum(dim=1)
    col_mass = A.sum(dim=0)
    term1 = (D_X.pow(2) @ row_mass).unsqueeze(1).expand_as(A)
    term2 = 2.0 * (D_X @ A @ D_Y)
    term3 = (D_Y.pow(2) @ col_mass).unsqueeze(0).expand_as(A)
    return 2.0 * (term1 - term2 + term3)


def _make_column_marginal(
    cost: Tensor,
    bpe_lengths: Optional[Tensor],
    column_marginal_type: str,
    bpe_col_floor: float,
    token_prior_sigma: float,
    token_prior_score_temp: float,
    token_prior_floor: float,
) -> Tensor:
    _, num_tokens = cost.shape
    if column_marginal_type == "bpe":
        if bpe_lengths is None:
            raise ValueError("bpe column marginal requires bpe_lengths")
        return _make_bpe_length_column_marginal(
            bpe_lengths.to(cost.device), floor=bpe_col_floor
        )
    if column_marginal_type == "acoustic":
        return _make_soft_column_marginal(
            cost,
            sigma=token_prior_sigma,
            score_temp=token_prior_score_temp,
            floor=token_prior_floor,
        )
    if column_marginal_type == "uniform":
        return _make_uniform_column_marginal(num_tokens, cost.device)
    raise ValueError(f"Unsupported column_marginal_type: {column_marginal_type}")


def vi_fgw_loss_v2(
    log_p_nonblank: Tensor,
    alpha: Tensor,
    labels: Tensor,
    acoustic_features: Tensor,
    bpe_lengths: Optional[Tensor] = None,
    column_marginal_type: str = "acoustic",
    alpha_smooth_mix: float = 0.1,
    bpe_col_floor: float = 0.05,
    token_prior_sigma: float = 0.15,
    token_prior_score_temp: float = 1.0,
    token_prior_floor: float = 0.05,
    eps: float = 0.3,
    iters: int = 30,
    beta_pos: float = 1.0,
    lambda_gw: float = 0.1,
    n_outer: int = 3,
    return_plan: bool = False,
) -> Tensor | Tuple[Tensor, Optional[Tensor]]:
    """VFTA OT loss whose detached transport plan includes an FGW term.

    The final objective remains ``<stop_gradient(plan), acoustic_cost>``. Thus
    lambda changes transport geometry while gradients reach the CTC nonblank
    classifier through the same path as the original VFTA OT objective.
    """
    if lambda_gw <= 0.0:
        return vi_ot_loss_v2(
            log_p_nonblank=log_p_nonblank,
            alpha=alpha,
            labels=labels,
            bpe_lengths=bpe_lengths,
            column_marginal_type=column_marginal_type,
            alpha_smooth_mix=alpha_smooth_mix,
            bpe_col_floor=bpe_col_floor,
            token_prior_sigma=token_prior_sigma,
            token_prior_score_temp=token_prior_score_temp,
            token_prior_floor=token_prior_floor,
            eps=eps,
            iters=iters,
            beta_pos=beta_pos,
            return_plan=return_plan,
        )

    log_probs = log_p_nonblank.float()
    num_frames, vocab_minus_blank = log_probs.shape
    num_tokens = labels.numel()
    if num_frames == 0 or num_tokens == 0:
        zero = log_probs.new_tensor(0.0)
        return (zero, None) if return_plan else zero
    if acoustic_features.size(0) != num_frames:
        raise ValueError(
            "acoustic_features must match the encoder time axis: "
            f"{acoustic_features.size(0)} != {num_frames}"
        )

    label_idx = (labels.long() - 1).clamp(
        min=0, max=vocab_minus_blank - 1
    )
    acoustic_cost = -log_probs[:, label_idx]
    acoustic_cost_detached = acoustic_cost.detach()
    detached_cost = acoustic_cost_detached
    pos_cost = _make_pos_cost(
        num_frames,
        num_tokens,
        device=log_probs.device,
        beta_pos=beta_pos,
    )
    if pos_cost is not None:
        detached_cost = detached_cost + pos_cost

    row_marginal = _make_alpha_row_marginal(
        alpha.detach().float(), smooth_mix=alpha_smooth_mix
    )
    col_marginal = _make_column_marginal(
        # Keep the acoustic token marginal identical to lambda=0 OT. The
        # positional and GW terms may alter the coupling, not its marginals.
        cost=acoustic_cost_detached,
        bpe_lengths=bpe_lengths,
        column_marginal_type=column_marginal_type,
        bpe_col_floor=bpe_col_floor,
        token_prior_sigma=token_prior_sigma,
        token_prior_score_temp=token_prior_score_temp,
        token_prior_floor=token_prior_floor,
    )

    with torch.no_grad():
        plan = _sinkhorn(
            row_marginal, col_marginal, detached_cost, eps=eps, iters=iters
        )
        acoustic_structure = _compute_dx(acoustic_features.detach())
        token_structure = _compute_dy(num_tokens, device=log_probs.device)
        cost_scale = detached_cost.std().clamp_min(1.0e-6)
        structural_weight = min(max(float(lambda_gw), 0.0), 1.0)
        for _ in range(int(n_outer)):
            gradient = _gw_gradient(
                acoustic_structure, token_structure, plan
            )
            gradient = gradient * (
                cost_scale / gradient.std().clamp_min(1.0e-6)
            )
            effective_cost = (
                (1.0 - structural_weight) * detached_cost
                + structural_weight * gradient
            )
            plan = _sinkhorn(
                row_marginal,
                col_marginal,
                effective_cost,
                eps=eps,
                iters=iters,
            )

    loss = (plan.detach() * acoustic_cost).sum()
    if return_plan:
        return loss, plan
    return loss


@torch.no_grad()
def _masked_std_batched(values: Tensor, mask: Tensor) -> Tensor:
    """Match torch.std() independently over each batch item's valid cells."""
    weights = mask.to(values.dtype)
    count = weights.sum(dim=(1, 2)).clamp_min(1.0)
    mean = (values * weights).sum(dim=(1, 2)) / count
    squared = ((values - mean[:, None, None]).pow(2) * weights).sum(dim=(1, 2))
    return (squared / (count - 1.0).clamp_min(1.0)).sqrt()


def vi_fgw_loss_v2_batched(
    log_p_nonblank: Tensor,
    alpha: Tensor,
    labels: Tensor,
    frame_lens: Tensor,
    label_lens: Tensor,
    acoustic_features: Tensor,
    bpe_lengths: Optional[Tensor] = None,
    column_marginal_type: str = "acoustic",
    alpha_smooth_mix: float = 0.1,
    bpe_col_floor: float = 0.05,
    token_prior_sigma: float = 0.15,
    token_prior_score_temp: float = 1.0,
    token_prior_floor: float = 0.05,
    eps: float = 0.3,
    iters: int = 30,
    beta_pos: float = 1.0,
    lambda_gw: float = 0.1,
    n_outer: int = 3,
    return_plan: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Vectorized version of :func:`vi_fgw_loss_v2` for padded batches."""
    if lambda_gw <= 0.0:
        return vi_ot_loss_v2_batched(
            log_p_nonblank=log_p_nonblank,
            alpha=alpha,
            labels=labels,
            frame_lens=frame_lens,
            label_lens=label_lens,
            bpe_lengths=bpe_lengths,
            column_marginal_type=column_marginal_type,
            alpha_smooth_mix=alpha_smooth_mix,
            bpe_col_floor=bpe_col_floor,
            token_prior_sigma=token_prior_sigma,
            token_prior_score_temp=token_prior_score_temp,
            token_prior_floor=token_prior_floor,
            eps=eps,
            iters=iters,
            beta_pos=beta_pos,
            return_plan=return_plan,
        )

    tiny = 1.0e-8
    lp = log_p_nonblank.float()
    batch_size, max_frames, vocab_minus_blank = lp.shape
    max_tokens = labels.size(1)
    device = lp.device
    if acoustic_features.shape[:2] != (batch_size, max_frames):
        raise ValueError(
            "acoustic_features must have shape [B,T,F] matching log_p_nonblank"
        )

    row_mask = torch.arange(max_frames, device=device)[None] < frame_lens[:, None]
    col_mask = torch.arange(max_tokens, device=device)[None] < label_lens[:, None]
    cell_mask = row_mask[:, :, None] & col_mask[:, None, :]
    label_idx = (labels.long() - 1).clamp(0, vocab_minus_blank - 1)
    acoustic_cost = -torch.gather(
        lp,
        2,
        label_idx[:, None, :].expand(batch_size, max_frames, max_tokens),
    )
    acoustic_cost_detached = acoustic_cost.detach()

    t_frac = _frac_positions(frame_lens, max_frames, device)
    u_frac = _frac_positions(label_lens, max_tokens, device)
    plan_cost = acoustic_cost_detached
    if beta_pos > 0:
        plan_cost = plan_cost + float(beta_pos) * (
            t_frac[:, :, None] - u_frac[:, None, :]
        ).pow(2)

    row_marginal = _masked_normalize(
        alpha.detach().float().clamp_min(tiny), row_mask, tiny
    )
    row_marginal = _mix_uniform_batched(
        row_marginal,
        row_mask,
        frame_lens,
        alpha_smooth_mix,
        tiny,
    )

    if column_marginal_type == "bpe":
        if bpe_lengths is None:
            raise ValueError("bpe column marginal requires bpe_lengths")
        col_marginal = _masked_normalize(
            bpe_lengths.float().clamp_min(1.0), col_mask, tiny
        )
        col_marginal = _mix_uniform_batched(
            col_marginal, col_mask, label_lens, bpe_col_floor, tiny
        )
    elif column_marginal_type == "uniform":
        col_marginal = col_mask.float() / label_lens.float().clamp_min(1.0)[:, None]
    elif column_marginal_type == "acoustic":
        temperature = max(float(token_prior_score_temp), tiny)
        sigma = max(float(token_prior_sigma), tiny)
        scores = (-acoustic_cost_detached / temperature).masked_fill(
            ~col_mask[:, None, :], _NEG
        )
        acoustic_scores = torch.softmax(scores, dim=2)
        position_mask = torch.exp(
            -0.5
            * ((t_frac[:, :, None] - u_frac[:, None, :]) / sigma).pow(2)
        )
        token_mass = (
            (position_mask * acoustic_scores)
            .masked_fill(~row_mask[:, :, None], 0.0)
            .sum(dim=1)
        )
        col_marginal = _masked_normalize(token_mass, col_mask, tiny)
        col_marginal = _mix_uniform_batched(
            col_marginal, col_mask, label_lens, token_prior_floor, tiny
        )
    else:
        raise ValueError(f"Unsupported column_marginal_type: {column_marginal_type}")

    def solve(cost: Tensor) -> Tensor:
        log_kernel = -cost / max(float(eps), tiny)
        log_kernel = log_kernel.masked_fill(~cell_mask, _NEG)
        log_a = torch.log(row_marginal.clamp_min(tiny)).masked_fill(
            ~row_mask, _NEG
        )
        log_b = torch.log(col_marginal.clamp_min(tiny)).masked_fill(
            ~col_mask, _NEG
        )
        log_plan = _sinkhorn_batched(log_a, log_b, log_kernel, iters)
        return torch.exp(log_plan).masked_fill(~cell_mask, 0.0)

    with torch.no_grad():
        plan = solve(plan_cost)
        normalized = F.normalize(acoustic_features.detach().float(), dim=-1)
        acoustic_structure = (
            1.0 - normalized @ normalized.transpose(1, 2)
        ).clamp_min(0.0)
        token_index = torch.arange(max_tokens, device=device, dtype=torch.float32)
        token_structure = (
            token_index[None, :, None] - token_index[None, None, :]
        ).abs() / label_lens.float().clamp_min(1.0)[:, None, None]
        cost_scale = _masked_std_batched(plan_cost, cell_mask).clamp_min(1.0e-6)
        structural_weight = min(max(float(lambda_gw), 0.0), 1.0)
        for _ in range(int(n_outer)):
            row_mass = plan.sum(dim=2)
            col_mass = plan.sum(dim=1)
            term1 = torch.bmm(
                acoustic_structure.pow(2), row_mass.unsqueeze(2)
            ).expand(-1, -1, max_tokens)
            term2 = 2.0 * torch.bmm(
                torch.bmm(acoustic_structure, plan), token_structure
            )
            term3 = torch.bmm(
                token_structure.pow(2), col_mass.unsqueeze(2)
            ).transpose(1, 2).expand(-1, max_frames, -1)
            gradient = 2.0 * (term1 - term2 + term3)
            gradient_scale = _masked_std_batched(gradient, cell_mask).clamp_min(
                1.0e-6
            )
            gradient = gradient * (
                cost_scale / gradient_scale
            )[:, None, None]
            effective_cost = (
                (1.0 - structural_weight) * plan_cost
                + structural_weight * gradient
            )
            plan = solve(effective_cost)

    loss = (plan.detach() * acoustic_cost).sum(dim=(1, 2))
    valid = (frame_lens > 0) & (label_lens > 0)
    loss = torch.where(valid, loss, loss.new_zeros(()))
    if return_plan:
        return loss, plan
    return loss
