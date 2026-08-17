"""
CTC + 1D Wasserstein Alignment Regularization
==============================================

Motivation
----------
Directly modifying CTC path scores (via OT tilting) is unstable because any
asymmetric tilt (e.g., only non-blank states) distorts the blank vs. non-blank
competition in the CTC dynamic programming recursion.

This module instead adds a *separate* 1D Wasserstein term alongside the
standard CTC loss:

    L = L_CTC  +  lambda_w1 * L_W1

L_W1 does not touch the CTC loss at all.  It pushes the model's per-token
temporal emission distributions to be roughly monotonically aligned, providing
a gradient that fights blank-peaky or unordered alignments.

Formulation
-----------
For each token position k (0 … N-1):

  1. Compute log-odds of token k vs blank at each frame t:

         z_k[t] = log p(y_k | x_t)  -  log p(blank | x_t)

     Blank-dominated frames yield z_k[t] << 0, so they get near-zero weight.
     This is how blank is handled implicitly — no explicit masking needed.

  2. Normalise over time (softmax-over-time):

         p_k[t] = exp(z_k[t] / tau) / sum_{t'} exp(z_k[t'] / tau)

     p_k is the model's belief about "when does token k get emitted?".

  3. Target: token k should be emitted around frame  t*_k = T * (k + 0.5) / N.
     The target is a step (Dirac delta in the continuous limit):

         Q_k[t] = 0  if  t < t*_k,   else  1      (target CDF)

  4. 1D Wasserstein  =  integral |CDF_empirical - CDF_target|:

         W1_k = (1/T) * sum_t | cumsum(p_k)[t]  -  Q_k[t] |

     This has an exact closed-form and is differentiable w.r.t. log_probs.

  5. Total alignment loss (mean over tokens and utterances):

         L_W1 = mean_k  W1_k

Gradient intuition
------------------
  - If the model emits token k too early  (CDF rises before Q_k):
    grad pushes z_k[t] DOWN for early frames, UP for late frames
    → model learns to delay token k emission

  - If the model emits token k too late  (CDF lags behind Q_k):
    opposite — model learns to advance token k emission

  - Blank frames: z_k[t] is low → p_k[t] ≈ 0 → effectively no gradient
    through blank-heavy frames, so blank probability is not artificially
    distorted.

Notes on repeated tokens
------------------------
When the same vocab ID appears twice in token_ids (e.g., Y = [a, b, a]),
both occurrences share the same temporal distribution p_k because softmax-over-t
of log_probs[:, vocab_id] cannot distinguish the first from the second occurrence.
The targets t*_1 and t*_5 (for first and second 'a') are different, so their
W1 terms may conflict.  For BPE transcripts repetitions are rare; in cases
where they matter, one can fall back to the CTC-posterior-based version
(see ctc_w1_with_posterior, not implemented here).
"""

from typing import Optional
import torch


def ctc_w1_alignment_loss(
    log_probs:   torch.Tensor,   # [T, V]  log-softmax output (has grad)
    token_ids:   torch.Tensor,   # [N]     non-blank target token ids
    blank_id:    int   = 0,
    temperature: float = 1.0,    # softmax-over-time sharpness
) -> torch.Tensor:               # scalar
    """
    1D Wasserstein alignment loss for one utterance.

    Parameters
    ----------
    log_probs   : [T, V] log-softmax probabilities from the encoder.
    token_ids   : [N] non-blank target token ids (int64).
    blank_id    : vocab index of the blank symbol.
    temperature : controls sharpness of the per-token temporal distribution.
                  Lower = sharper (puts more weight on the peak frame).
                  Typical range: 0.5 – 2.0.

    Returns
    -------
    Scalar W1 alignment loss, differentiable w.r.t. log_probs.
    """
    T = log_probs.shape[0]
    N = int(token_ids.numel())
    if N == 0 or T < 2:
        return log_probs.new_tensor(0.0)

    device = log_probs.device
    dtype  = log_probs.dtype

    # ── Per-token log-odds vs blank  ──────────────────────────────────────
    # token_lp: [T, N]  log p(y_k | x_t)
    # blank_lp: [T, 1]  log p(blank | x_t)
    # log_odds: [T, N]  high when token k is preferred over blank at frame t
    token_lp = log_probs[:, token_ids]                       # [T, N]
    blank_lp = log_probs[:, blank_id].unsqueeze(1)           # [T, 1]
    log_odds  = (token_lp - blank_lp) / temperature          # [T, N]

    # ── Temporal distribution p_k  ────────────────────────────────────────
    # Softmax over the TIME dimension (dim=0), not over vocabulary.
    # p_k[t] ≈ 0 for blank-dominated frames (log_odds << 0 there).
    p_k = log_odds.softmax(dim=0)                            # [T, N]

    # ── Empirical CDF  ────────────────────────────────────────────────────
    P_k = p_k.cumsum(dim=0)                                  # [T, N]

    # ── Target CDF: step at t*_k = T * (k+0.5) / N  ─────────────────────
    t_stars = torch.linspace(0.5 / N, 1.0 - 0.5 / N, N,
                             device=device, dtype=dtype) * T  # [N]
    t_grid  = torch.arange(T, device=device, dtype=dtype)     # [T]
    # Q_k[t, k] = 1 if t >= t*_k, else 0
    Q_k = (t_grid.unsqueeze(1) >= t_stars.unsqueeze(0)).to(dtype)  # [T, N]

    # ── 1D Wasserstein = (1/T) * sum_t |P_k[t] - Q_k[t]|  ───────────────
    w1_per_token = (P_k - Q_k).abs().mean(dim=0)             # [N]  (mean over T)

    return w1_per_token.mean()                                # scalar


def ctc_w1_alignment_loss_batch(
    log_probs_list: list,         # list of [T_i, V]  (has grad)
    token_ids_list: list,         # list of [N_i]
    blank_id:       int   = 0,
    temperature:    float = 1.0,
) -> torch.Tensor:                # [B] per-utterance W1 losses
    """
    Batched version: one W1 loss per utterance.

    Each call to ctc_w1_alignment_loss is O(T_i * N_i) and fully vectorized;
    the Python loop here is over utterances only (no T-loop inside).

    Returns [B] per-utterance losses (same order as the input lists).
    """
    losses = [
        ctc_w1_alignment_loss(lp, ids, blank_id=blank_id, temperature=temperature)
        for lp, ids in zip(log_probs_list, token_ids_list)
    ]
    return torch.stack(losses)   # [B]
