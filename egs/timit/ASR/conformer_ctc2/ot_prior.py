import torch
import torch.nn.functional as F


def _sinkhorn(a, b, C, eps=0.1, iters=30):
    """
    a: [T], b: [U], C: [T,U]  (all on same device)
    returns P: [T,U]

    We solve Sinkhorn in the log domain for stability. This is important when
    training under autocast/fp16 because exp(-C / eps) can easily underflow.
    """
    tiny = 1e-8
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
    P = torch.exp(log_P)
    return P


@torch.no_grad()
def _make_pos_cost(T, U, device, beta_pos):
    if beta_pos <= 0:
        return None
    tpos = torch.linspace(0, 1, T, device=device).unsqueeze(1)  # [T,1]
    upos = torch.linspace(0, 1, U, device=device).unsqueeze(0)  # [1,U]
    return beta_pos * (tpos - upos).pow(2)  # [T,U]


@torch.no_grad()
def _make_soft_column_marginal(
    D_detach: torch.Tensor,
    sigma: float = 0.15,
    score_temp: float = 1.0,
    floor: float = 0.05,
) -> torch.Tensor:
    """
    Build a non-uniform token prior b[u] from a soft diagonal mask and
    acoustic matching scores.

    Args:
      D_detach:
        Detached OT cost with shape [T, U].
      sigma:
        Width of the diagonal soft mask in normalized time/token space.
      score_temp:
        Temperature for converting negative OT cost into soft acoustic scores.
      floor:
        Mix a small uniform prior to keep every token reachable.
    Returns:
      A probability vector with shape [U].
    """
    T, U = D_detach.shape
    device = D_detach.device
    tiny = 1e-8
    if U == 0:
        return D_detach.new_zeros((0,))

    if sigma <= 0:
        return torch.full((U,), 1.0 / U, device=device)

    tpos = torch.linspace(0, 1, T, device=device).unsqueeze(1)  # [T,1]
    upos = torch.linspace(0, 1, U, device=device).unsqueeze(0)  # [1,U]
    pos_mask = torch.exp(-0.5 * ((tpos - upos) / max(float(sigma), tiny)).pow(2))

    temp = max(float(score_temp), tiny)
    acoustic_scores = torch.softmax(-D_detach / temp, dim=1)
    token_mass = (pos_mask * acoustic_scores).sum(dim=0)

    b = token_mass / token_mass.sum().clamp_min(tiny)
    if floor > 0:
        uniform = torch.full_like(b, 1.0 / U)
        mix = min(max(float(floor), 0.0), 1.0)
        b = (1.0 - mix) * b + mix * uniform
        b = b / b.sum().clamp_min(tiny)
    return b


@torch.no_grad()
def _make_soft_row_marginal(
    log_probs: torch.Tensor,
    blank_id: int = 0,
    uniform_mix: float = 0.0,
) -> torch.Tensor:
    """Build a soft frame prior from non-blank probabilities over all frames.

    Each frame participates in OT. Frames with higher non-blank probability
    receive more transport mass, while `uniform_mix` prevents the row marginal
    from collapsing onto only a few frames.
    """
    T = log_probs.size(0)
    if T == 0:
        return log_probs.new_zeros((0,))

    tiny = 1e-8
    nonblank = (1.0 - log_probs[:, blank_id].exp()).clamp_min(tiny)
    a = nonblank / nonblank.sum().clamp_min(tiny)

    if uniform_mix > 0:
        uniform = torch.full_like(a, 1.0 / T)
        mix = min(max(float(uniform_mix), 0.0), 1.0)
        a = (1.0 - mix) * a + mix * uniform
        a = a / a.sum().clamp_min(tiny)

    return a


def ot_prior_loss_from_logprobs(
    log_probs,          # [T,V] log softmax
    labels,             # [U] token ids (NO blank)
    bpe_lengths=None,   # Optional [U] BPE character lengths for column marginal
    blank_id=0,
    tau=0.1,            # uniform mix weight for soft blank row marginal
    eps=0.5,
    iters=20,
    beta_pos=1.0,       # diagonal bias strength
    token_prior_sigma=0.15,
    token_prior_score_temp=1.0,
    token_prior_floor=0.05,
    return_plan: bool = False,
):
    """
    Returns a scalar OT prior loss for one utterance:
      L_ot = <P*, D>,   D_{t,u}=-log p(y_u|t)

    OT is computed using stop-grad cost (handled outside) by passing
    cost_detached=True pattern; we implement as:
      P* from D_detach, loss uses D (non-detached).
    """

    log_probs = log_probs.float()
    device = log_probs.device
    T, V = log_probs.shape
    U = labels.numel()
    if U == 0 or T == 0:
        if return_plan:
            return log_probs.new_tensor(0.0), None, None
        return log_probs.new_tensor(0.0)

    # D (non-detached) and D_detach (for OT)
    # D: [T,U]
    D = -log_probs.gather(
        dim=1, index=labels.unsqueeze(0).expand(T, U)
    )
    D_detach = D.detach()

    # add diagonal/monotonic bias
    pos_cost = _make_pos_cost(T, U, device=device, beta_pos=beta_pos)
    if pos_cost is not None:
        D_detach = D_detach + pos_cost
        D = D + pos_cost  # also apply to training signal

    # Every frame stays in OT. The soft blank mask enters through the row
    # marginal instead of a hard active-frame subset.
    a = _make_soft_row_marginal(
        log_probs=log_probs, blank_id=blank_id, uniform_mix=tau
    )
    if bpe_lengths is None:
        b = _make_soft_column_marginal(
            D_detach=D_detach,
            sigma=token_prior_sigma,
            score_temp=token_prior_score_temp,
            floor=token_prior_floor,
        )
    else:
        b = _make_bpe_length_column_marginal(
            bpe_lengths=bpe_lengths.to(device),
            floor=token_prior_floor,
        )

    # solve OT on detached cost
    P = _sinkhorn(a, b, D_detach, eps=eps, iters=iters)  # [T,U]

    # weighted NLL using NON-detached D
    loss = (P * D).sum()
    if return_plan:
        keep = torch.ones((T,), device=device, dtype=torch.bool)
        return loss, P, keep
    return loss


# ── FGW helpers ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_dx(acoustic_features: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance matrix from acoustic features.
    acoustic_features: [T, D]
    returns: [T, T]
    """
    normed = F.normalize(acoustic_features.float(), dim=-1)
    return (1.0 - normed @ normed.T).clamp(min=0.0)


@torch.no_grad()
def _compute_dy(U: int, device) -> torch.Tensor:
    """Linear position distance D^Y[u,u'] = |u-u'| / U.
    returns: [U, U]
    """
    idx = torch.arange(U, device=device, dtype=torch.float32)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() / U


@torch.no_grad()
def _gw_gradient(D_X: torch.Tensor, D_Y: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Gradient of GW energy wrt A (used in conditional gradient outer loop).
    G[t,u] = 2 * Σ_{t'u'} A[t'u'] (D_X[t,t'] - D_Y[u,u'])²
    D_X: [T,T], D_Y: [U,U], A: [T,U]
    returns: [T,U]  (no gradient tracked)
    """
    r = A.sum(dim=1)          # [T]  row marginals of current A
    c = A.sum(dim=0)          # [U]  col marginals
    DX2 = D_X.pow(2)
    DY2 = D_Y.pow(2)
    term1 = (DX2 @ r).unsqueeze(1).expand_as(A)   # [T,U]
    term2 = 2.0 * (D_X @ A @ D_Y)                  # [T,U]
    term3 = (DY2 @ c).unsqueeze(0).expand_as(A)   # [T,U]
    return 2.0 * (term1 - term2 + term3)


def fgw_ot_loss_from_logprobs(
    log_probs,              # [T, V]  log softmax
    acoustic_features,      # [T, D]  raw/spec acoustic features for D^X
    labels,                 # [U]  token ids (no blank)
    blank_id: int = 0,
    tau: float = 0.1,
    eps: float = 0.5,
    iters: int = 20,
    beta_pos: float = 1.0,
    token_prior_sigma: float = 0.15,
    token_prior_score_temp: float = 1.0,
    token_prior_floor: float = 0.05,
    lambda_gw: float = 0.0,
    n_outer: int = 3,       # Frank-Wolfe outer iterations
    return_plan: bool = False,
):
    """OT prior loss with optional FGW structural term (Frank-Wolfe).

    When lambda_gw=0, reduces to the standard entropic OT with soft column
    marginal (same as ot_prior_loss_from_logprobs).

    When lambda_gw>0, runs n_outer iterations of conditional gradient (Frank-Wolfe)
    to include the GW pairwise structure cost: raw acoustic frame-pair distances
    (D^X) should match token position-pair distances (D^Y).

    The FGW plan A* is computed entirely on stop-grad costs; θ-gradients flow
    only through the acoustic cost C in the final loss = <A*, C>.

    G is scale-normalised to match C before combining so that the plan entropy
    does not collapse when lambda_gw > 0.
    """
    log_probs = log_probs.float()
    device = log_probs.device
    T = log_probs.shape[0]
    U = labels.numel()

    if U == 0 or T == 0:
        return (log_probs.new_tensor(0.0), None, None) if return_plan else log_probs.new_tensor(0.0)

    # acoustic cost C[t,u] = -log p(y_u | t)  — carries θ gradient
    C = -log_probs.gather(1, labels.unsqueeze(0).expand(T, U))

    pos_cost = _make_pos_cost(T, U, device=device, beta_pos=beta_pos)
    C_detach = C.detach()
    if pos_cost is not None:
        C_detach = C_detach + pos_cost

    # The whole sequence participates in OT; blank confidence shapes the
    # row marginal softly instead of pruning frames out of the plan.
    a = _make_soft_row_marginal(
        log_probs=log_probs, blank_id=blank_id, uniform_mix=tau
    )
    b = _make_soft_column_marginal(
        D_detach=C_detach,
        sigma=token_prior_sigma,
        score_temp=token_prior_score_temp,
        floor=token_prior_floor,
    )

    if lambda_gw <= 0.0:
        # pure OT — no GW term
        A = _sinkhorn(a, b, C_detach, eps=eps, iters=iters)
    else:
        # FGW via conditional gradient (Frank-Wolfe), all stop-grad
        with torch.no_grad():
            D_X = _compute_dx(acoustic_features.detach())  # [T, T]
            D_Y = _compute_dy(U, device=device)      # [U, U]
            c_scale = C_detach.std().clamp(min=1e-6)

            # initialise with pure OT solution
            A = _sinkhorn(a, b, C_detach, eps=eps, iters=iters)

            for _ in range(n_outer):
                G = _gw_gradient(D_X, D_Y, A)       # [T', U]
                # Normalise G to match C's scale so the GW term does not
                # inflate C_eff and collapse the plan's entropy.
                G_normed = G * (c_scale / G.std().clamp(min=1e-6))
                C_eff = (1.0 - lambda_gw) * C_detach + lambda_gw * G_normed
                A = _sinkhorn(a, b, C_eff, eps=eps, iters=iters)

    # loss uses NON-detached C so θ gets gradients; A is stop-grad
    loss_C = C if pos_cost is None else C + pos_cost
    loss = (A * loss_C).sum()

    if return_plan:
        keep = torch.ones((T,), device=device, dtype=torch.bool)
        return loss, A, keep
    return loss


# ── VI-OT helpers (full VI framework with blank gate) ─────────────────────────

def _make_alpha_row_marginal(alpha: torch.Tensor, smooth_mix: float = 0.1) -> torch.Tensor:
    """
    Row marginal driven by blank gate posterior alpha_t.

    alpha:      [T]  per-frame non-blank probabilities from BlankGateHead
    smooth_mix: uniform mix weight; prevents a sharply peaked alpha from
                making OT trivial (all mass pre-assigned by the gate).

    Returns a probability vector [T] that sums to 1.
    """
    tiny = 1e-8
    T = alpha.numel()
    a = alpha.clamp_min(tiny)
    a = a / a.sum().clamp_min(tiny)
    if smooth_mix > 0 and T > 0:
        uniform = torch.full_like(a, 1.0 / T)
        mix = min(max(float(smooth_mix), 0.0), 1.0)
        a = (1.0 - mix) * a + mix * uniform
        a = a / a.sum().clamp_min(tiny)
    return a


def _make_bpe_length_column_marginal(
    bpe_lengths: torch.Tensor,
    floor: float = 0.05,
) -> torch.Tensor:
    """
    Column marginal proportional to BPE token character lengths.

    bpe_lengths: [U]  float32 character counts per token (computed in training
                      script via sp.id_to_piece — kept out of this file to avoid
                      sentencepiece dependency here).
    floor:       mix weight for uniform floor so every token stays reachable.

    Longer BPE tokens (more characters) receive more frame mass — linguistically
    motivated prior that is completely θ-independent (no circular gradient).
    """
    tiny = 1e-8
    U = bpe_lengths.numel()
    if U == 0:
        return bpe_lengths.new_zeros((0,))
    b = bpe_lengths.clamp_min(1.0)
    b = b / b.sum().clamp_min(tiny)
    if floor > 0:
        uniform = torch.full_like(b, 1.0 / U)
        mix = min(max(float(floor), 0.0), 1.0)
        b = (1.0 - mix) * b + mix * uniform
        b = b / b.sum().clamp_min(tiny)
    return b


def vi_ot_loss_from_logprobs(
    log_probs: torch.Tensor,        # [T, V]  gated log-softmax
    alpha: torch.Tensor,            # [T]     blank gate posterior (can be detached)
    labels: torch.Tensor,           # [U]     token ids (no blank)
    bpe_lengths: torch.Tensor,      # [U]     float32 BPE char lengths
    blank_id: int = 0,
    alpha_smooth_mix: float = 0.1,
    bpe_col_floor: float = 0.05,
    eps: float = 0.5,
    iters: int = 30,
    beta_pos: float = 1.0,
    return_plan: bool = False,
):
    """
    OT alignment loss for the full VI framework.

    Differences from ot_prior_loss_from_logprobs:
      - Row marginal: alpha-based (_make_alpha_row_marginal) instead of model blank probs
      - Column marginal: BPE char lengths (_make_bpe_length_column_marginal) instead of
        acoustic scores (which are derived from model output and create circular gradients)

    The OT plan A* is solved on stop-grad costs; θ-gradients flow through C = -log p_gated.

    Returns scalar loss (and optionally the plan + frame-keep mask).
    """
    log_probs = log_probs.float()
    device = log_probs.device
    T, V = log_probs.shape
    U = labels.numel()
    if U == 0 or T == 0:
        if return_plan:
            return log_probs.new_tensor(0.0), None, None
        return log_probs.new_tensor(0.0)

    # Cost C[t,u] = -log p_gated(y_u | t)  — carries θ-gradient
    C = -log_probs.gather(1, labels.unsqueeze(0).expand(T, U))
    C_detach = C.detach()

    pos_cost = _make_pos_cost(T, U, device=device, beta_pos=beta_pos)
    if pos_cost is not None:
        C_detach = C_detach + pos_cost

    # Row marginal from blank gate (detach alpha so OT plan is stop-grad w.r.t. psi)
    a = _make_alpha_row_marginal(alpha.detach().float(), smooth_mix=alpha_smooth_mix)

    # Column marginal from BPE lengths (completely θ-independent)
    b = _make_bpe_length_column_marginal(bpe_lengths.to(device), floor=bpe_col_floor)

    # Solve OT on stop-grad cost; plan A* is a constant for gradient purposes
    P = _sinkhorn(a, b, C_detach, eps=eps, iters=iters)   # [T, U]

    # Weighted NLL: gradient flows through C (not P)
    loss_C = C if pos_cost is None else C + pos_cost
    loss = (P * loss_C).sum()

    if return_plan:
        keep = torch.ones((T,), device=device, dtype=torch.bool)
        return loss, P, keep
    return loss
