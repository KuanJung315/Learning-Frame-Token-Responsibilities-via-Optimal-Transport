"""CTC forward-backward occupancy and the faithful posterior-consistency loss.

Implements term (C) of the VFTA lower bound as written in the paper:
    A_theta(t, u) = P(y_u is emitted at frame t | X, Y)
computed by alpha/beta recursions over the standard CTC trellis
(2U+1 states, blank-interleaved), and
    L_align = - sum_{t,u} sg[A*_{t,u}] log Abar_theta(t, u)
where both the transport plan A* and the occupancy are normalized to unit
mass over the [T, U] grid.  Gradients flow through the occupancy into the
blank-aware emissions (gate + non-blank classifier + encoder); the plan is
a detached structured target, exactly as in the plan-cost variant.

All recursions are batched ([B, T, 2U+1]) and autograd-friendly.  Padded
frames are handled by forcing them to emit blank with probability one, so
they extend the final blank state without changing the total likelihood
and carry zero token occupancy.
"""

from typing import Tuple

import torch
from torch import Tensor

_NEG = -1.0e30  # finite -inf sentinel: keeps logsumexp gradients NaN-free


def ctc_fb_occupancy(
    log_probs: Tensor,  # [B, T, V] log emission probs, blank = index 0
    targets: Tensor,  # [B, U] padded token ids (>= 1 for real tokens)
    input_lengths: Tensor,  # [B]
    target_lengths: Tensor,  # [B]
) -> Tuple[Tensor, Tensor]:
    """Returns (occupancy [B, T, U], log_Z [B]).

    occupancy[b, t, u] = P(token position u emitted at frame t | X, Y),
    zero on padded frames/positions.  log_Z is the CTC log-likelihood
    (matches torch.nn.functional.ctc_loss up to sign).
    """
    lp = log_probs.float()
    B, T, V = lp.shape
    U = targets.size(1)
    S = 2 * U + 1
    device = lp.device

    # State labels: even states emit blank (0), odd state 2u+1 emits y_u.
    z = torch.zeros(B, S, dtype=torch.long, device=device)
    z[:, 1::2] = targets.long().clamp(min=0, max=V - 1)

    s_idx = torch.arange(S, device=device)
    state_valid = s_idx.unsqueeze(0) < (2 * target_lengths + 1).unsqueeze(1)  # [B,S]
    t_idx = torch.arange(T, device=device)
    frame_valid = t_idx.unsqueeze(0) < input_lengths.unsqueeze(1)  # [B,T]

    # Per-state emission log-probs; padded frames emit blank with prob 1.
    lp_z = torch.gather(lp, 2, z.unsqueeze(1).expand(B, T, S))  # [B,T,S]
    pad_emit = torch.where(
        (s_idx % 2 == 0), lp.new_zeros(()), lp.new_full((), _NEG)
    )  # [S]
    lp_z = torch.where(frame_valid.unsqueeze(2), lp_z, pad_emit.view(1, 1, S))
    lp_z = lp_z.masked_fill(~state_valid.unsqueeze(1), _NEG)

    # Skip transition s-2 -> s allowed for odd s >= 3 with distinct tokens.
    allow_skip = torch.zeros(B, S, dtype=torch.bool, device=device)
    if U >= 2:
        allow_skip[:, 3::2] = targets[:, 1:] != targets[:, :-1]
    allow_skip &= state_valid

    def _shift(x: Tensor, n: int) -> Tensor:
        return torch.cat([x.new_full((B, n), _NEG), x[:, : S - n]], dim=1)

    # Forward.
    la = lp.new_full((B, S), _NEG)
    la[:, 0] = lp_z[:, 0, 0]
    if S > 1:
        la[:, 1] = lp_z[:, 0, 1]
    alphas = [la]
    for t in range(1, T):
        skip = _shift(la, 2).masked_fill(~allow_skip, _NEG)
        la = torch.logsumexp(torch.stack([la, _shift(la, 1), skip]), dim=0)
        la = la + lp_z[:, t]
        alphas.append(la)
    alpha = torch.stack(alphas, dim=1)  # [B,T,S]

    # Final states: 2*tl (last blank) and 2*tl - 1 (last token).
    e1 = (2 * target_lengths).long().clamp(min=0)
    e2 = (2 * target_lengths - 1).long().clamp(min=0)
    last = alpha[:, T - 1]
    log_z = torch.logsumexp(
        torch.stack([last.gather(1, e1.unsqueeze(1)), last.gather(1, e2.unsqueeze(1))]),
        dim=0,
    ).squeeze(1)

    # Backward: beta_t(s) = log P(emissions t+1..T-1 | state s at t).
    lb = lp.new_full((B, S), _NEG)
    lb = lb.scatter(1, e1.unsqueeze(1), 0.0)
    lb = lb.scatter(1, e2.unsqueeze(1), 0.0)
    betas = [lb]
    for t in range(T - 2, -1, -1):
        nxt = lb + lp_z[:, t + 1]  # arrive at same s
        down1 = torch.cat([nxt[:, 1:], nxt.new_full((B, 1), _NEG)], dim=1)
        skip_ok = torch.cat(
            [allow_skip[:, 2:], allow_skip.new_zeros(B, 2)], dim=1
        )
        down2 = torch.cat([nxt[:, 2:], nxt.new_full((B, 2), _NEG)], dim=1)
        down2 = down2.masked_fill(~skip_ok, _NEG)
        lb = torch.logsumexp(torch.stack([nxt, down1, down2]), dim=0)
        betas.append(lb)
    betas.reverse()
    beta = torch.stack(betas, dim=1)  # [B,T,S]

    gamma = alpha + beta - log_z.view(B, 1, 1)
    occ = torch.exp(gamma[:, :, 1::2])  # token states -> [B,T,U]

    u_valid = torch.arange(U, device=device).unsqueeze(0) < target_lengths.unsqueeze(1)
    occ = occ * frame_valid.unsqueeze(2) * u_valid.unsqueeze(1)
    return occ, log_z


def fb_posterior_consistency_loss(
    plan: Tensor,  # [B, T, U] transport plan (treated as detached target)
    gated_log_probs: Tensor,  # [B, T, V] blank-aware emission log-probs
    targets: Tensor,  # [B, U]
    input_lengths: Tensor,  # [B]
    target_lengths: Tensor,  # [B]
    eps: float = 1.0e-8,
) -> Tensor:
    """L_align = -<sg[A*], log Abar_theta>, per-utterance vector [B].

    Both views are normalized to unit mass over the [T, U] grid before the
    cross-entropy (the plan already sums to one by construction).  Utterances
    with a non-finite CTC likelihood contribute zero loss.
    """
    occ, log_z = ctc_fb_occupancy(
        gated_log_probs, targets, input_lengths, target_lengths
    )
    denom = occ.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
    log_occ_norm = occ.clamp_min(eps).log() - denom.log()
    loss = -(plan.detach() * log_occ_norm).sum(dim=(1, 2))
    valid = (
        torch.isfinite(log_z)
        & (input_lengths > 0)
        & (target_lengths > 0)
        & torch.isfinite(loss)
    )
    return torch.where(valid, loss, loss.new_zeros(()))
