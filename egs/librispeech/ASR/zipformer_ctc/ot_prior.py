import torch


def _sinkhorn(a, b, C, eps=0.1, iters=50):
    """
    a: [T], b: [U], C: [T,U]  (all on same device)
    returns P: [T,U]

    Log-domain Sinkhorn for numerical stability under fp16/autocast.
    C should be detached before calling; gradient flows through a (soft marginal).
    """
    tiny = 1e-8
    eps = max(float(eps), tiny)

    log_a = torch.log(a.clamp_min(tiny))
    log_b = torch.log(b.clamp_min(tiny))
    log_K = -C / eps

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)

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


def ot_prior_loss_from_logprobs(
    log_probs,          # [T,V] log softmax
    labels,             # [U] token ids (NO blank)
    blank_id=0,
    eps=0.5,
    iters=40,
    beta_pos=1.0,
    lambda_ent=0.0,     # >0: explicit entropy regularization to prevent collapse
    return_plan=False,
):
    """
    OT alignment loss with self-consistent soft marginals.

    Both frame marginal a and token marginal b are derived from the model's
    current output, requiring no external duration information:

      a_t = (1 - p_blank_t) / Z          [how much non-blank mass frame t carries]
      b_u = sum_t a_t * p(y_u | t)       [how much mass token u naturally attracts]

    b reflects the model's current belief about per-token duration:
    tokens the model spends more frames on get larger b_u.  This removes
    the uniform-b mismatch while staying fully differentiable.

    Loss = <P_detach, D_nll> / T  -  lambda_ent * H(P)
    """
    log_probs = log_probs.float()
    device = log_probs.device
    T, V = log_probs.shape
    U = labels.numel()

    if U == 0 or T < 2:
        if return_plan:
            return log_probs.new_tensor(0.0), None, None
        return log_probs.new_tensor(0.0)

    # ------------------------------------------------------------------
    # Frame marginal a: weight frames by non-blank probability.
    # Differentiable through log_probs[:, blank_id].
    # ------------------------------------------------------------------
    p_blank = log_probs[:, blank_id].exp()
    non_blank = (1.0 - p_blank).clamp(min=1e-6)
    a = non_blank / non_blank.sum()          # [T], differentiable

    # ------------------------------------------------------------------
    # Token marginal b: project frame mass onto token space.
    #   b_u = sum_t  a_t * p(y_u | t)
    # Reflects model's current predicted duration for each token.
    # Detached so b only shapes the OT structure, not the gradient.
    # ------------------------------------------------------------------
    with torch.no_grad():
        p_tokens = log_probs[:, labels].exp()          # [T, U]
        b_unnorm = (a.detach().unsqueeze(1) * p_tokens).sum(dim=0)  # [U]
        b = (b_unnorm / b_unnorm.sum().clamp_min(1e-8)).clamp_min(1e-8)
        b = b / b.sum()                                # re-normalise for safety

    # ------------------------------------------------------------------
    # NLL cost matrix [T, U]
    # Detached for Sinkhorn (stable P); gradient enters through D below.
    # ------------------------------------------------------------------
    D = -log_probs.gather(
        dim=1, index=labels.unsqueeze(0).expand(T, U)
    )  # [T, U], has grad

    pos_cost = _make_pos_cost(T, U, device, beta_pos)
    C_detach = (D.detach() + pos_cost) if pos_cost is not None else D.detach()

    # ------------------------------------------------------------------
    # Sinkhorn on detached cost.
    # P has gradient through a (soft marginal) but not through C.
    # ------------------------------------------------------------------
    P = _sinkhorn(a, b, C_detach, eps=eps, iters=iters)  # [T, U]

    # ------------------------------------------------------------------
    # Transport loss: reweighted NLL.
    # Gradient flows through D (token log-probs) using stop-grad P.
    # ------------------------------------------------------------------
    transport_loss = (P.detach() * D).sum() / T

    # ------------------------------------------------------------------
    # Entropy regularisation: maximise H(P) to prevent alignment collapse.
    # Gradient flows through P → a → log_probs[:, blank_id].
    # Larger lambda_ent → more spread-out alignment (EnCTC-like effect).
    # ------------------------------------------------------------------
    if lambda_ent > 0.0:
        H_P = -(P * P.clamp_min(1e-8).log()).sum()
        loss = transport_loss - lambda_ent * H_P
    else:
        loss = transport_loss

    if return_plan:
        return loss, P.detach(), None
    return loss
