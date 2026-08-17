"""
OT-Reweighted CTC Loss  (v2 — state-level tilt + batched DP)
=============================================================

Formulation
-----------
  L = -log  sum_{pi in B^{-1}(Y)}  prod_t  exp( log_emit_tilt[t, pi_t] )

where:
  Extended label   tilde_Y = (blank, y_1, blank, y_2, ..., blank, y_N, blank)
                   U = 2N+1 states,  s_u in {blank, y_0, ..., y_{N-1}}

  Tilted emission (per CTC state, not per vocabulary class):
    log_emit_tilt[t, u_blank] = log p(blank | x_t)              (no tilt)
    log_emit_tilt[t, u_tok_k] = log p(y_k   | x_t)              t not in M
    log_emit_tilt[t, u_tok_k] = log p(y_k   | x_t)              t in M
                               + lambda * log(gamma*[t_M, k] + delta)

  gamma* = Sinkhorn OT plan over [T_M, N]:
    cost C[t_M, k] = -log p(y_k | x_t_M)  (+ positional bias)
    M = { t : p(non-blank | x_t) > tau }  (hard frame mask)
    marginals: a = uniform(T_M),  b = uniform(N)
    stop-gradient on gamma*

Key differences from v1 (vocabulary-tilt + F.ctc_loss)
-------------------------------------------------------
v1 BUG-A  scatter_add onto vocabulary dim loses position info.
          When token y_k appears multiple times, ALL positions' tilts were
          summed into the same class — CTC needs per-POSITION (per-STATE)
          tilts to correctly prefer one occurrence over another.
v1 BUG-B  F.ctc_loss re-normalises via implicit softmax at every frame,
          computing  -log sum_pi prod_t softmax(tilde_l)[pi_t].
          This normalisation adds a gradient term -sum_t log Z_t that
          partially cancels the OT signal and cannot be tuned away.
v1 BUG-C  Blank states receive zero tilt but F.ctc_loss competes blank vs
          non-blank under the tilted distribution — the competition is
          broken because blank probability is unchanged while non-blank
          probabilities are boosted by OT.

v2 fixes
--------
- OT tilt applied at EXTENDED STATE level [T, U], not vocabulary [T, V].
  Each state u has its own tilt entry: blank states always 0, non-blank
  states get position-specific tilt from gamma*[t_M, k].
- Custom forward DP (no implicit normalisation).  The loss is exactly
  -log sum_pi prod_t exp(log_emit_tilt[t, pi_t]).
- Batched DP: a single Python loop over T (not over utterances × T),
  processing all B utterances in parallel each step.
  Speed: O(T_max) Python steps vs O(B × T_max) in the per-utterance approach.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Log-domain Sinkhorn  (stop-gradient; unchanged from v1)
# ---------------------------------------------------------------------------

def _sinkhorn_log(
    a: torch.Tensor,   # [T_M]
    b: torch.Tensor,   # [N]
    C: torch.Tensor,   # [T_M, N]
    eps: float = 0.1,
    iters: int = 50,
) -> torch.Tensor:     # [T_M, N]  coupling, no grad
    """Log-domain Sinkhorn.  Always called inside torch.no_grad()."""
    eps = max(float(eps), 1e-9)
    tiny = 1e-9
    a = a.float().clamp_min(tiny)
    b = b.float().clamp_min(tiny)
    C = C.float()
    log_a = a.log()
    log_b = b.log()
    log_K = -C / eps
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)
    return torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))


# ---------------------------------------------------------------------------
# NaN-safe logaddexp  (fixes -inf + -inf → NaN in backward)
# ---------------------------------------------------------------------------

class _SafeLogAddExp(torch.autograd.Function):
    """
    logaddexp with NaN-safe backward.

    Standard backward: d/da = exp(a - out).  When a = b = out = -inf,
    exp(-inf - (-inf)) = exp(nan) = nan.  This propagates through the DP
    to produce NaN gradients even at reachable states.

    Fix: replace NaN with 0 (unreachable states have zero gradient).
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.logaddexp(a, b)
        ctx.save_for_backward(a, b, out)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        a, b, out = ctx.saved_tensors
        da = (grad * (a - out).exp_()).nan_to_num_(nan=0.0)
        db = (grad * (b - out).exp_()).nan_to_num_(nan=0.0)
        return da, db


def _safe_logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _SafeLogAddExp.apply(a, b)


# ---------------------------------------------------------------------------
# Per-utterance: OT plan + tilted [T, U] emission matrix
# ---------------------------------------------------------------------------

def _prepare_one(
    log_probs:       torch.Tensor,           # [T, V], float32, has grad
    token_ids:       torch.Tensor,           # [N], int64
    blank_id:        int,
    tau:             float,
    eps:             float,
    sinkhorn_iters:  int,
    lambda_ot:       float,
    beta_pos:        float,
    delta:           float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Build the tilted emission matrix for one utterance.

    Returns
    -------
    log_emit   : [T, U=2N+1]  tilted log emissions (gradient flows through)
    can_skip   : [U]          bool — states that allow the skip-2 CTC transition
    gamma_star : [T_M, N]     OT coupling (no grad), or None
    keep       : [T]          bool frame mask  (p_nonblank > tau)
    """
    T, V = log_probs.shape
    N = int(token_ids.numel())
    U = 2 * N + 1
    device = log_probs.device

    # ── Extended label index array ────────────────────────────────────────
    # label_ids[2k]   = blank_id          (blank states)
    # label_ids[2k+1] = token_ids[k]      (non-blank states)
    label_ids = torch.full((U,), blank_id, dtype=torch.long, device=device)
    if N > 0:
        label_ids[1::2] = token_ids

    # ── Gather log p(s_u | x_t) for each extended state ──────────────────
    # Advanced indexing: log_emit[t, u] = log_probs[t, label_ids[u]]
    # Gradient flows back to log_probs[:, blank_id] and log_probs[:, token_ids]
    log_emit = log_probs[:, label_ids]   # [T, U], differentiable ✓

    # ── can_skip[u]: True if state u allows the skip-2 CTC transition ────
    # Condition: u is odd, u >= 3, token at u != token two non-blanks before
    can_skip = torch.zeros(U, dtype=torch.bool, device=device)
    if N >= 2:
        odd_u  = torch.arange(3, U, 2, device=device)   # [N-1]: states 3,5,...
        idx_c  = (odd_u - 1) // 2
        idx_p  = idx_c - 1
        can_skip[odd_u] = token_ids[idx_c] != token_ids[idx_p]

    # ── Hard frame mask M ─────────────────────────────────────────────────
    with torch.no_grad():
        p_blank    = log_probs[:, blank_id].exp()
        p_nonblank = (1.0 - p_blank).clamp_(0.0, 1.0)
        keep = p_nonblank > tau                          # [T] bool

    T_M = int(keep.sum().item())

    # ── OT plan (stop-gradient) ───────────────────────────────────────────
    gamma_star: Optional[torch.Tensor] = None
    if T_M >= 2 and N >= 1 and lambda_ot > 0.0:
        with torch.no_grad():
            lp_m = log_probs[keep]                           # [T_M, V]
            C    = -lp_m[:, token_ids]                       # [T_M, N]  semantic cost
            if beta_pos > 0.0:
                t_pos = torch.linspace(0, 1, T_M, device=device).unsqueeze(1)
                u_pos = torch.linspace(0, 1, N,   device=device).unsqueeze(0)
                C = C + beta_pos * (t_pos - u_pos).pow(2)   # + positional bias
            a = C.new_full((T_M,), 1.0 / T_M)
            b = C.new_full((N,),   1.0 / N)
            gamma_star = _sinkhorn_log(a, b, C, eps=eps, iters=sinkhorn_iters)
            # gamma_star: [T_M, N], requires_grad=False ✓

    # ── Add OT tilt to non-blank states at masked frames ─────────────────
    # log_emit[:, 1::2] are the N non-blank state columns
    # Only masked frames (t in M) get the tilt; blank columns unchanged
    if gamma_star is not None:
        ot_tilt   = lambda_ot * torch.log(gamma_star + delta)   # [T_M, N], no grad
        full_tilt = log_emit.new_zeros(T, U)                     # [T, U],   no grad
        full_tilt[keep, 1::2] = ot_tilt                          # scatter to odd cols
        log_emit  = log_emit + full_tilt                         # [T, U], grad ✓

    return log_emit, can_skip, gamma_star, keep


# ---------------------------------------------------------------------------
# Batched CTC forward DP
# Single Python loop over T_max; all B utterances processed in parallel.
# ---------------------------------------------------------------------------

def _ctc_dp_batch(
    log_emit_batch: torch.Tensor,   # [B, T_max, U_max], grad
    can_skip_batch: torch.Tensor,   # [B, U_max], bool (padded cols = False)
    T_lens:         torch.Tensor,   # [B] int — actual frame count per utterance
    U_lens:         torch.Tensor,   # [B] int — actual state count (2N+1)
) -> torch.Tensor:                  # [B] log sequence scores, grad
    """
    CTC forward recursion on the tilted emission matrix, batched.

    Boundary conditions (0-indexed):
      alpha[0, 0]   = log_emit[0, 0]       (first blank)
      alpha[0, 1]   = log_emit[0, 1]       (first token, if U >= 2)
      alpha[0, u>1] = -inf

    Transition (for each t >= 1, each state u):
      pred[u]  = logsumexp(alpha[u], alpha[u-1])                 always
               + logsumexp(…, alpha[u-2])   if can_skip[u]
      new_alpha[u] = pred[u] + log_emit[t, u]

    Terminal:
      log_score = logsumexp(alpha[T-1, U-1], alpha[T-1, U-2])
    """
    B, T_max, U_max = log_emit_batch.shape
    device = log_emit_batch.device
    dtype  = log_emit_batch.dtype

    # ── Initialise alpha ──────────────────────────────────────────────────
    alpha = log_emit_batch.new_full((B, U_max), _NEG_INF)

    # State 0 (first blank): valid for all utterances
    alpha[:, 0] = log_emit_batch[:, 0, 0]

    # State 1 (first token): valid only if U >= 2 (i.e. N >= 1)
    has_token = U_lens >= 2                              # [B] bool
    if has_token.any():
        alpha[has_token, 1] = log_emit_batch[has_token, 0, 1]

    pad1 = alpha.new_full((B, 1), _NEG_INF)
    pad2 = alpha.new_full((B, 2), _NEG_INF)

    # ── Recursion t = 1 … T_max-1 ────────────────────────────────────────
    for t in range(1, T_max):
        # valid_t[b]: True if frame t is within the utterance length
        valid_t = (t < T_lens).unsqueeze(1)              # [B, 1]

        # alpha[u-1] and alpha[u-2] via left-pad
        p1 = torch.cat([pad1, alpha[:, :-1]], dim=1)     # [B, U_max]
        p2 = torch.cat([pad2, alpha[:, :-2]], dim=1)     # [B, U_max]

        s12  = _safe_logaddexp(alpha, p1)                 # self + u-1
        s123 = _safe_logaddexp(s12,   p2)                 # + u-2

        pred      = torch.where(can_skip_batch, s123, s12)         # [B, U_max]
        new_alpha = pred + log_emit_batch[:, t]                     # [B, U_max]

        # Keep old alpha for frames beyond the utterance length
        alpha = torch.where(valid_t, new_alpha, alpha)

    # ── Termination ───────────────────────────────────────────────────────
    B_idx    = torch.arange(B, device=device)
    last_idx = (U_lens - 1).clamp(min=0)                 # [B]  U-1
    prev_idx = (U_lens - 2).clamp(min=0)                 # [B]  U-2  (min 0 for safety)

    log_score = _safe_logaddexp(
        alpha[B_idx, last_idx],
        alpha[B_idx, prev_idx],
    )   # [B]

    # Edge case: U==1 means both indices are 0 → logaddexp(x,x) = x + log2, wrong.
    # But U==1 (N==0) is filtered before calling this function, so this is safe.

    return log_score


# ---------------------------------------------------------------------------
# CTC forward-backward: returns log_score AND path entropy H(Z|X,Y)
# ---------------------------------------------------------------------------

def _ctc_fb_batch(
    log_emit_batch: torch.Tensor,   # [B, T_max, U_max], grad
    can_skip_batch: torch.Tensor,   # [B, U_max], bool
    T_lens:         torch.Tensor,   # [B] int
    U_lens:         torch.Tensor,   # [B] int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CTC forward AND backward pass.  Returns (log_score [B], path_entropy [B]).

    Path entropy  H(Z|X,Y) ≈ -(1/T) sum_t sum_u  A[t,u] * log A[t,u]

    where  A[t,u] = exp(alpha[t,u] + beta[t,u] - log_score)  is the CTC
    occupancy posterior — the probability that state u is active at frame t
    given the transcript and acoustic features.

    This is the information-theoretically correct entropy to maximise in order
    to prevent CTC path collapse: it measures uncertainty in the ALIGNMENT
    space (U=2N+1 CTC states per frame), not the VOCABULARY space.

    Gradient of path_entropy w.r.t. log_emit flows through both alpha and beta.
    Add  -lambda_H * path_entropy  to the total loss to maximise entropy
    (subtract from loss = maximise).
    """
    B, T_max, U_max = log_emit_batch.shape
    device = log_emit_batch.device
    dtype  = log_emit_batch.dtype

    pad1 = log_emit_batch.new_full((B, 1), _NEG_INF)
    pad2 = log_emit_batch.new_full((B, 2), _NEG_INF)

    # ── Forward pass: store full alpha history ────────────────────────────
    alpha_all = log_emit_batch.new_full((B, T_max, U_max), _NEG_INF)
    alpha = log_emit_batch.new_full((B, U_max), _NEG_INF)
    alpha[:, 0] = log_emit_batch[:, 0, 0]
    has_token = U_lens >= 2
    if has_token.any():
        alpha[has_token, 1] = log_emit_batch[has_token, 0, 1]
    alpha_all[:, 0] = alpha

    for t in range(1, T_max):
        valid_t = (t < T_lens).unsqueeze(1)
        p1 = torch.cat([pad1, alpha[:, :-1]], dim=1)
        p2 = torch.cat([pad2, alpha[:, :-2]], dim=1)
        s12  = _safe_logaddexp(alpha, p1)
        s123 = _safe_logaddexp(s12,   p2)
        pred      = torch.where(can_skip_batch, s123, s12)
        new_alpha = pred + log_emit_batch[:, t]
        alpha = torch.where(valid_t, new_alpha, alpha)
        alpha_all[:, t] = alpha

    B_idx    = torch.arange(B, device=device)
    last_idx = (U_lens - 1).clamp(min=0)
    prev_idx = (U_lens - 2).clamp(min=0)
    log_score = _safe_logaddexp(alpha[B_idx, last_idx], alpha[B_idx, prev_idx])

    # ── Backward pass: store full beta history ────────────────────────────
    # beta[t, u] = log probability of completing the path from state u at t
    # Boundary: beta[T-1, U-1] = beta[T-1, U-2] = 0  (log 1), rest = -inf
    beta_all = log_emit_batch.new_full((B, T_max, U_max), _NEG_INF)
    beta = log_emit_batch.new_full((B, U_max), _NEG_INF)
    # Set terminal states for each utterance
    beta[B_idx, last_idx] = 0.0
    # Only set prev_idx if it differs from last_idx (i.e., U >= 2)
    different = last_idx != prev_idx
    if different.any():
        beta[B_idx[different], prev_idx[different]] = 0.0
    beta_all[:, T_max - 1] = beta

    # Backward transitions: who can follow state u at t+1?
    # State v follows u if u in N(v), i.e., v == u, v == u+1, or v == u+2 (if can_skip[v])
    # Equivalently: beta[t, u] = sum_{v in successors(u)}  exp(log_emit[t+1, v]) * beta[t+1, v]
    # Successor of u: v == u (self), v == u+1 (advance), v == u+2 (skip, if can_skip[v])
    pad1_r = log_emit_batch.new_full((B, 1), _NEG_INF)
    pad2_r = log_emit_batch.new_full((B, 2), _NEG_INF)

    for t in range(T_max - 2, -1, -1):
        valid_t = (t < T_lens - 1).unsqueeze(1)           # frame t+1 must exist

        # Future contribution: log_emit[t+1, v] + beta[t+1, v]
        future = log_emit_batch[:, t + 1] + beta           # [B, U_max]

        # beta[t, u] = logsumexp over successors:
        # successor u (self-loop): future[u]
        # successor u+1 (advance): future[u+1]  → shift future left by 1
        # successor u+2 (skip):    future[u+2], gated by can_skip[u+2]
        f_self   = future
        f_next1  = torch.cat([future[:, 1:], pad1_r], dim=1)    # future[u+1]
        f_next2  = torch.cat([future[:, 2:], pad2_r], dim=1)    # future[u+2]

        s12_b  = _safe_logaddexp(f_self, f_next1)
        # can_skip for u+2: shift can_skip_batch left by 2
        skip_shifted = torch.cat([can_skip_batch[:, 2:],
                                   torch.zeros(B, 2, dtype=torch.bool, device=device)],
                                  dim=1)
        s123_b = _safe_logaddexp(s12_b, f_next2)
        new_beta = torch.where(skip_shifted, s123_b, s12_b)
        beta = torch.where(valid_t, new_beta, beta)
        beta_all[:, t] = beta

    # ── Occupancy posterior  A[t, u] = exp(alpha + beta - log_score) ─────
    # Only compute for valid (t, u) positions; clamp log_score for safety
    log_score_bc = log_score.unsqueeze(1).unsqueeze(2)           # [B, 1, 1]
    log_A = alpha_all + beta_all - log_score_bc                  # [B, T_max, U_max]

    # Mask out invalid frames (t >= T_lens) and invalid states (u >= U_lens)
    t_mask = (torch.arange(T_max, device=device).unsqueeze(0) <
              T_lens.unsqueeze(1))                                # [B, T_max]
    u_mask = (torch.arange(U_max, device=device).unsqueeze(0) <
              U_lens.unsqueeze(1))                                # [B, U_max]
    valid_mask = t_mask.unsqueeze(2) & u_mask.unsqueeze(1)       # [B, T_max, U_max]
    log_A = log_A.masked_fill(~valid_mask, _NEG_INF)

    # ── H(Z|X,Y) per utterance ────────────────────────────────────────────
    # A[t,:] sums to 1 for each valid t.
    # Entropy = -sum_u  A * log A  with convention 0*log(0) = 0.
    #
    # NaN-safe backward: the gradient of -exp(x)*x at x = -inf is
    #   d/dx[-exp(x)*x] = -exp(x)*(1+x) → 0 * (-inf) = NaN.
    # Fix: gate the computation with a threshold mask so the autograd
    # never sees x = -inf in the active branch.
    log_A_thresh = -50.0                                         # exp(-50) ≈ 2e-22
    active = log_A > log_A_thresh                                # [B, T_max, U_max]
    A_active  = log_A.clamp(min=log_A_thresh).exp()             # gradient path
    neg_entr  = A_active * log_A.clamp(min=log_A_thresh)        # A * log A
    H_tu = torch.where(active, -neg_entr,
                       torch.zeros_like(neg_entr))               # [B, T_max, U_max]
    H_t  = H_tu.sum(dim=2)                                      # [B, T_max]
    T_lens_f = T_lens.float().clamp(min=1)
    path_entropy = H_t.sum(dim=1) / T_lens_f                    # [B]

    return log_score, path_entropy


# ---------------------------------------------------------------------------
# Public API: batch loss
# ---------------------------------------------------------------------------

def ot_reweighted_ctc_loss_batch(
    log_probs_list:  List[torch.Tensor],    # each [T_i, V], has grad
    token_ids_list:  List[torch.Tensor],    # each [N_i], int64
    blank_id:        int   = 0,
    tau:             float = 0.1,
    eps:             float = 0.1,
    sinkhorn_iters:  int   = 30,
    lambda_ot:       float = 1.0,
    beta_pos:        float = 1.0,
    delta:           float = 1e-6,
    lambda_H:        float = 0.0,
    return_plans:    bool  = False,
) -> "torch.Tensor | Tuple[torch.Tensor, list]":
    """
    OT-Reweighted CTC loss for a batch of utterances.

    Parameters
    ----------
    log_probs_list  : list of [T_i, V] log-softmax tensors (one per utterance).
    token_ids_list  : list of [N_i] non-blank target token ids (int64).
    blank_id        : vocabulary index of the blank symbol.
    tau             : non-blank probability threshold for the hard frame mask M.
    eps             : Sinkhorn entropy regularisation.
    sinkhorn_iters  : Sinkhorn iteration count.
    lambda_ot       : OT tilt strength.  lambda_ot=0 → standard CTC.
    beta_pos        : positional diagonal bias added to the OT cost matrix.
    delta           : stability floor in log(gamma* + delta).
    lambda_H        : CTC path entropy regularisation weight.
                      When > 0, runs the full forward-backward DP and adds
                      -lambda_H * H(Z|X,Y) to each utterance loss.
                      H(Z|X,Y) is the CTC alignment entropy — measures
                      uncertainty in which CTC path is taken.
                      Set to 0 (default) to skip the backward pass.
    return_plans    : if True, also return a list of (gamma_star, keep) per utt.

    Returns
    -------
    losses  : [B] per-utterance losses  -log S(Y|X) - lambda_H * H(Z|X,Y).
    plans   : (only when return_plans=True) list of (gamma_star or None, keep).
    """
    B = len(log_probs_list)
    assert B > 0

    device = log_probs_list[0].device

    # ── Per-utterance: Sinkhorn (cheap, tiny matrices) + build [T, U] ────
    log_emit_list: List[torch.Tensor] = []
    can_skip_list: List[torch.Tensor] = []
    T_lens_list:   List[int]          = []
    U_lens_list:   List[int]          = []
    plans_list:    list               = []
    degenerate:    List[bool]         = []   # True → skip DP, loss = 0

    for log_probs, token_ids in zip(log_probs_list, token_ids_list):
        T = log_probs.shape[0]
        N = int(token_ids.numel())

        if N == 0 or T == 0:
            # Degenerate utterance: no valid CTC path → loss = 0
            log_emit_list.append(log_probs.float().new_zeros(1, 1))
            can_skip_list.append(torch.zeros(1, dtype=torch.bool, device=device))
            T_lens_list.append(1)
            U_lens_list.append(1)
            degenerate.append(True)
            if return_plans:
                plans_list.append((None,
                                   torch.zeros(T, dtype=torch.bool, device=device)))
            continue

        log_emit, can_skip, gamma_star, keep = _prepare_one(
            log_probs.float(), token_ids, blank_id,
            tau, eps, sinkhorn_iters, lambda_ot, beta_pos, delta,
        )
        log_emit_list.append(log_emit)       # [T, U]
        can_skip_list.append(can_skip)       # [U]
        T_lens_list.append(T)
        U_lens_list.append(2 * N + 1)
        degenerate.append(False)
        if return_plans:
            plans_list.append((gamma_star, keep))

    # ── Pad to [B, T_max, U_max] ─────────────────────────────────────────
    T_max = max(T_lens_list)
    U_max = max(U_lens_list)

    padded_emit: List[torch.Tensor] = []
    padded_skip: List[torch.Tensor] = []

    for log_emit, can_skip, T_i, U_i in zip(
            log_emit_list, can_skip_list, T_lens_list, U_lens_list):
        # F.pad is differentiable; padded positions get -inf (no contribution)
        emit_p = F.pad(log_emit, (0, U_max - U_i, 0, T_max - T_i),
                       value=_NEG_INF)                   # [T_max, U_max]
        skip_p = F.pad(can_skip.float(), (0, U_max - U_i)).bool()  # [U_max]
        padded_emit.append(emit_p)
        padded_skip.append(skip_p)

    log_emit_batch = torch.stack(padded_emit, dim=0)     # [B, T_max, U_max]
    can_skip_batch = torch.stack(padded_skip, dim=0)     # [B, U_max]
    T_lens = torch.tensor(T_lens_list, device=device)
    U_lens = torch.tensor(U_lens_list, device=device)

    # ── Single batched DP ─────────────────────────────────────────────────
    # Use forward-backward if entropy regularisation is requested
    if lambda_H > 0.0:
        log_scores, path_entropy = _ctc_fb_batch(
            log_emit_batch, can_skip_batch, T_lens, U_lens)
    else:
        log_scores   = _ctc_dp_batch(log_emit_batch, can_skip_batch, T_lens, U_lens)
        path_entropy = None

    # loss = -log S(Y|X)  - lambda_H * H(Z|X,Y)
    # Adding -lambda_H * H maximises path entropy (prevents alignment collapse)
    losses = -log_scores
    if path_entropy is not None:
        losses = losses - lambda_H * path_entropy   # [B]

    # Infeasible alignments (T < U-1): loss would be +inf → set to 0
    losses = losses.masked_fill(~torch.isfinite(losses), 0.0)

    # Degenerate utterances: force loss = 0
    if any(degenerate):
        degen_mask = torch.tensor(degenerate, device=device)
        losses = losses.masked_fill(degen_mask, 0.0)

    if return_plans:
        return losses, plans_list
    return losses
