from typing import Optional

import torch
from torch import Tensor


def _sinkhorn(a: Tensor, b: Tensor, C: Tensor, eps: float = 0.3, iters: int = 30) -> Tensor:
    tiny = 1.0e-8
    eps = max(float(eps), tiny)

    log_a = torch.log(a.clamp_min(tiny))
    log_b = torch.log(b.clamp_min(tiny))
    log_K = -C / eps

    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K.transpose(0, 1) + log_u.unsqueeze(0), dim=1)

    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    return torch.exp(log_P)


def _make_alpha_row_marginal(alpha: Tensor, smooth_mix: float = 0.1) -> Tensor:
    tiny = 1.0e-8
    T = alpha.numel()
    if T == 0:
        return alpha.new_zeros((0,))

    a = alpha.clamp_min(tiny)
    a = a / a.sum().clamp_min(tiny)
    if smooth_mix > 0:
        uniform = torch.full_like(a, 1.0 / T)
        mix = min(max(float(smooth_mix), 0.0), 1.0)
        a = (1.0 - mix) * a + mix * uniform
        a = a / a.sum().clamp_min(tiny)
    return a


def _make_bpe_length_column_marginal(bpe_lengths: Tensor, floor: float = 0.05) -> Tensor:
    tiny = 1.0e-8
    U = bpe_lengths.numel()
    if U == 0:
        return bpe_lengths.new_zeros((0,))

    b = bpe_lengths.float().clamp_min(1.0)
    b = b / b.sum().clamp_min(tiny)
    if floor > 0:
        uniform = torch.full_like(b, 1.0 / U)
        mix = min(max(float(floor), 0.0), 1.0)
        b = (1.0 - mix) * b + mix * uniform
        b = b / b.sum().clamp_min(tiny)
    return b


def _make_uniform_column_marginal(U: int, device: torch.device) -> Tensor:
    if U == 0:
        return torch.zeros((0,), device=device)
    return torch.full((U,), 1.0 / U, device=device)


def _make_soft_column_marginal(
    C_detached: Tensor,
    sigma: float = 0.15,
    score_temp: float = 1.0,
    floor: float = 0.05,
) -> Tensor:
    """
    Previous OT token prior: estimate the column marginal from a detached
    diagonal/acoustic score map instead of BPE token lengths.
    """
    tiny = 1.0e-8
    T, U = C_detached.shape
    device = C_detached.device
    if U == 0:
        return C_detached.new_zeros((0,))
    if T == 0 or sigma <= 0:
        return _make_uniform_column_marginal(U, device)

    t_pos = torch.linspace(0, 1, T, device=device).unsqueeze(1)
    u_pos = torch.linspace(0, 1, U, device=device).unsqueeze(0)
    pos_mask = torch.exp(-0.5 * ((t_pos - u_pos) / max(float(sigma), tiny)).pow(2))

    temp = max(float(score_temp), tiny)
    acoustic_scores = torch.softmax(-C_detached / temp, dim=1)
    token_mass = (pos_mask * acoustic_scores).sum(dim=0)

    b = token_mass / token_mass.sum().clamp_min(tiny)
    if floor > 0:
        uniform = torch.full_like(b, 1.0 / U)
        mix = min(max(float(floor), 0.0), 1.0)
        b = (1.0 - mix) * b + mix * uniform
        b = b / b.sum().clamp_min(tiny)
    return b


def vi_ot_loss_v2(
    log_p_nonblank: Tensor,
    alpha: Tensor,
    labels: Tensor,
    bpe_lengths: Optional[Tensor] = None,
    column_marginal_type: str = "bpe",
    alpha_smooth_mix: float = 0.1,
    bpe_col_floor: float = 0.05,
    token_prior_sigma: float = 0.15,
    token_prior_score_temp: float = 1.0,
    token_prior_floor: float = 0.05,
    eps: float = 0.3,
    iters: int = 30,
    beta_pos: float = 1.0,
    return_plan: bool = False,
):
    """
    OT alignment loss for VarCTC v2.

    Cost uses only the V-1 non-blank classifier:
      C[t, u] = -log_p_nonblank[t, labels[u] - 1]

    The blank gate contributes only the detached row marginal.  Sinkhorn also
    runs on detached costs, so gradients flow through the final <P*, C> term to
    the non-blank head and encoder, not through the gate or the OT plan.
    """
    log_p_nonblank = log_p_nonblank.float()
    T, vocab_minus_blank = log_p_nonblank.shape
    U = labels.numel()
    device = log_p_nonblank.device

    if U == 0 or T == 0:
        if return_plan:
            return log_p_nonblank.new_tensor(0.0), None
        return log_p_nonblank.new_tensor(0.0)

    label_idx = (labels.long() - 1).clamp(min=0, max=vocab_minus_blank - 1)
    C = -log_p_nonblank[:, label_idx]
    C_detached = C.detach()

    pos_cost = None
    if beta_pos > 0:
        t_pos = torch.linspace(0, 1, T, device=device)
        u_pos = torch.linspace(0, 1, U, device=device)
        pos_cost = beta_pos * (t_pos.unsqueeze(1) - u_pos.unsqueeze(0)).pow(2)
        C_detached = C_detached + pos_cost

    a = _make_alpha_row_marginal(
        alpha.detach().float(),
        smooth_mix=alpha_smooth_mix,
    )
    if column_marginal_type == "bpe":
        if bpe_lengths is None:
            raise ValueError("bpe column marginal requires bpe_lengths")
        b = _make_bpe_length_column_marginal(
            bpe_lengths.to(device),
            floor=bpe_col_floor,
        )
    elif column_marginal_type == "acoustic":
        b = _make_soft_column_marginal(
            C_detached,
            sigma=token_prior_sigma,
            score_temp=token_prior_score_temp,
            floor=token_prior_floor,
        )
    elif column_marginal_type == "uniform":
        b = _make_uniform_column_marginal(U, device)
    else:
        raise ValueError(f"Unsupported column_marginal_type: {column_marginal_type}")

    P = _sinkhorn(a, b, C_detached, eps=eps, iters=iters)
    loss = (P.detach() * C).sum()

    if return_plan:
        return loss, P
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Batched (vectorized) OT — same math as the per-utterance vi_ot_loss_v2 above,
# but all B utterances are padded into [B, T_max, U_max] and solved with a single
# batched Sinkhorn.  Padding is handled with a NEG sentinel in log-space so the
# padded rows/cols contribute ~0 mass and never affect the valid region.  Result
# matches the per-utterance loop to float precision (see test_varctc_v2.py).
# ──────────────────────────────────────────────────────────────────────────────

_NEG = -1.0e9


def _masked_normalize(x: Tensor, mask: Tensor, tiny: float = 1.0e-8) -> Tensor:
    x = x.masked_fill(~mask, 0.0)
    return x / x.sum(dim=1, keepdim=True).clamp_min(tiny)


def _mix_uniform_batched(
    b: Tensor, mask: Tensor, lengths: Tensor, mix: float, tiny: float = 1.0e-8
) -> Tensor:
    """(1 - mix) * b + mix * uniform over each row's valid entries, renormalized.

    `b` is assumed already normalized per row over its valid entries.
    """
    mix = min(max(float(mix), 0.0), 1.0)
    if mix <= 0.0:
        return b
    uniform = mask.float() / lengths.float().clamp_min(1.0).unsqueeze(1)
    b = (1.0 - mix) * b + mix * uniform
    return _masked_normalize(b, mask, tiny=tiny)


def _frac_positions(lengths: Tensor, max_len: int, device: torch.device) -> Tensor:
    """Per-row linspace(0, 1, len_i) padded to max_len.  linspace(0,1,1) == [0]."""
    idx = torch.arange(max_len, device=device).float()
    den = (lengths.float() - 1.0).clamp_min(1.0).unsqueeze(1)
    return idx.unsqueeze(0) / den


def _sinkhorn_batched(
    log_a: Tensor,  # [B, T]    (NEG on padded rows)
    log_b: Tensor,  # [B, U]    (NEG on padded cols)
    log_K: Tensor,  # [B, T, U] (NEG on padded rows/cols)
    iters: int,
) -> Tensor:
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(int(iters)):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        log_v = log_b - torch.logsumexp(
            log_K.transpose(1, 2) + log_u.unsqueeze(1), dim=2
        )
    return log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)  # log_P [B, T, U]


def vi_ot_loss_v2_batched(
    log_p_nonblank: Tensor,  # [B, T, V-1]
    alpha: Tensor,  # [B, T]
    labels: Tensor,  # [B, U]  (long; padding entries ignored via label_lens)
    frame_lens: Tensor,  # [B]
    label_lens: Tensor,  # [B]
    bpe_lengths: Optional[Tensor] = None,  # [B, U] float
    column_marginal_type: str = "acoustic",
    alpha_smooth_mix: float = 0.1,
    bpe_col_floor: float = 0.05,
    token_prior_sigma: float = 0.15,
    token_prior_score_temp: float = 1.0,
    token_prior_floor: float = 0.05,
    eps: float = 0.3,
    iters: int = 30,
    beta_pos: float = 1.0,
    return_plan: bool = False,
    differentiable_plan: bool = False,
):
    """Batched VarCTC-v2 OT loss.  Returns a per-utterance loss vector [B],
    or (loss, plan [B, T, U]) when return_plan is set.

    Numerically equivalent to stacking per-utterance :func:`vi_ot_loss_v2` calls;
    gradients flow only through the final <P.detach(), C> term to the non-blank
    head, exactly as in the per-utterance path.
    """
    tiny = 1.0e-8
    lp = log_p_nonblank.float()
    B, T, Vm1 = lp.shape
    U = labels.shape[1]
    device = lp.device

    row_mask = torch.arange(T, device=device)[None, :] < frame_lens[:, None]  # [B,T]
    col_mask = torch.arange(U, device=device)[None, :] < label_lens[:, None]  # [B,U]

    # Cost C[b,t,u] = -log_p_nonblank[b, t, labels[b,u] - 1]
    label_idx = (labels.long() - 1).clamp(0, Vm1 - 1)
    C = -torch.gather(lp, 2, label_idx.unsqueeze(1).expand(B, T, U))  # [B,T,U]

    C_acoustic_detached = C.detach()
    C_plan = C if differentiable_plan else C_acoustic_detached
    t_frac = u_frac = None
    if beta_pos > 0:
        t_frac = _frac_positions(frame_lens, T, device)
        u_frac = _frac_positions(label_lens, U, device)
        pos_cost = beta_pos * (t_frac[:, :, None] - u_frac[:, None, :]).pow(2)
        C_plan = C_plan + pos_cost

    # The normal plan is a fixed teacher. ``differentiable_plan`` remains only
    # for backwards compatibility with old diagnostics and must not be used by
    # plan-to-CTC consistency training.
    alpha_for_plan = alpha.float() if differentiable_plan else alpha.detach().float()
    a = _masked_normalize(alpha_for_plan.clamp_min(tiny), row_mask, tiny)
    a = _mix_uniform_batched(a, row_mask, frame_lens, alpha_smooth_mix, tiny)

    # Column marginal.
    if column_marginal_type == "bpe":
        if bpe_lengths is None:
            raise ValueError("bpe column marginal requires bpe_lengths")
        b = _masked_normalize(bpe_lengths.float().clamp_min(1.0), col_mask, tiny)
        b = _mix_uniform_batched(b, col_mask, label_lens, bpe_col_floor, tiny)
    elif column_marginal_type == "uniform":
        b = col_mask.float() / label_lens.float().clamp_min(1.0).unsqueeze(1)
    elif column_marginal_type == "acoustic":
        temp = max(float(token_prior_score_temp), tiny)
        sigma = max(float(token_prior_sigma), tiny)
        if t_frac is None:
            t_frac = _frac_positions(frame_lens, T, device)
            u_frac = _frac_positions(label_lens, U, device)
        # Preserve the established VFTA semantics: the "acoustic" token
        # marginal is estimated from the same detached acoustic+position cost
        # used by Sinkhorn.  This also keeps the batched training path exactly
        # aligned with vi_ot_loss_v2 and the per-utterance FGW/eval path.
        scores = (-C_plan.detach() / temp).masked_fill(
            ~col_mask[:, None, :], _NEG
        )
        acoustic_scores = torch.softmax(scores, dim=2)
        pos_mask = torch.exp(
            -0.5 * ((t_frac[:, :, None] - u_frac[:, None, :]) / sigma).pow(2)
        )
        token_mass = (
            (pos_mask * acoustic_scores)
            .masked_fill(~row_mask[:, :, None], 0.0)
            .sum(dim=1)
        )  # [B,U]
        b = _masked_normalize(token_mass, col_mask, tiny)
        b = _mix_uniform_batched(b, col_mask, label_lens, token_prior_floor, tiny)
    else:
        raise ValueError(f"Unsupported column_marginal_type: {column_marginal_type}")

    # Batched Sinkhorn on the teacher cost (with NEG-masked padding).
    log_K = -C_plan / max(float(eps), tiny)
    log_K = log_K.masked_fill(~row_mask[:, :, None], _NEG)
    log_K = log_K.masked_fill(~col_mask[:, None, :], _NEG)
    log_a = torch.log(a.clamp_min(tiny)).masked_fill(~row_mask, _NEG)
    log_b = torch.log(b.clamp_min(tiny)).masked_fill(~col_mask, _NEG)

    log_P = _sinkhorn_batched(log_a, log_b, log_K, iters)
    P = torch.exp(log_P)
    P = P.masked_fill(~row_mask[:, :, None], 0.0).masked_fill(~col_mask[:, None, :], 0.0)

    loss = (P.detach() * C).sum(dim=(1, 2))  # [B], grad flows through C only
    valid = (frame_lens > 0) & (label_lens > 0)
    loss = torch.where(valid, loss, loss.new_zeros(()))
    if return_plan:
        return loss, P
    return loss
