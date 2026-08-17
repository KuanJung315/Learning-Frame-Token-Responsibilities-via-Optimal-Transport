from __future__ import annotations

import torch
from torch import Tensor


_NEG = -1.0e30


def ctc_token_occupancy_batched(
    log_probs: Tensor,
    labels: Tensor,
    frame_lens: Tensor,
    label_lens: Tensor,
    blank_id: int = 0,
) -> Tensor:
    """Return differentiable CTC token-state occupancies of shape [B, T, U].

    Only non-blank token states from the standard blank-interleaved CTC graph
    are returned. Repeated labels remain separate transcript positions because
    the last axis indexes graph states, not vocabulary symbols.

    Autograd is intentionally enabled: the detached OT/FGW plan is the teacher
    and this CTC occupancy is the student that receives the temporal matching
    gradient.
    """
    lp = log_probs.float()
    batch_size, num_frames, _ = lp.shape
    num_tokens = labels.size(1)
    device = lp.device

    if num_tokens == 0 or num_frames == 0:
        return lp.new_zeros((batch_size, num_frames, num_tokens))

    state_count = 2 * num_tokens + 1
    state_idx = torch.arange(state_count, device=device)
    token_state = state_idx % 2 == 1

    expanded = torch.full(
        (batch_size, state_count),
        int(blank_id),
        dtype=torch.long,
        device=device,
    )
    expanded[:, 1::2] = labels
    state_lens = 2 * label_lens + 1
    state_mask = state_idx.unsqueeze(0) < state_lens.unsqueeze(1)

    emit = torch.gather(
        lp,
        dim=2,
        index=expanded.unsqueeze(1).expand(batch_size, num_frames, state_count),
    )
    emit = emit.masked_fill(~state_mask.unsqueeze(1), _NEG)

    can_skip = token_state.unsqueeze(0).expand(batch_size, state_count).clone()
    if state_count > 2:
        can_skip[:, 2:] &= expanded[:, 2:] != expanded[:, :-2]
    can_skip &= state_mask

    neg1 = lp.new_full((batch_size, 1), _NEG)
    neg2 = lp.new_full((batch_size, 2), _NEG)

    init_mask = (state_idx.unsqueeze(0) == 0).expand(batch_size, -1)
    init_mask = init_mask | (
        (state_idx.unsqueeze(0) == 1) & (label_lens > 0).unsqueeze(1)
    )
    alpha = torch.where(init_mask, emit[:, 0], emit[:, 0].new_full((), _NEG))
    alpha_steps = [alpha]
    for t in range(1, num_frames):
        prev1 = torch.cat([neg1, alpha[:, :-1]], dim=1)
        prev2 = torch.cat([neg2, alpha[:, :-2]], dim=1)
        predecessor = torch.logaddexp(alpha, prev1)
        predecessor = torch.where(
            can_skip,
            torch.logaddexp(predecessor, prev2),
            predecessor,
        )
        new_alpha = predecessor + emit[:, t]
        alpha = torch.where((t < frame_lens).unsqueeze(1), new_alpha, alpha)
        alpha_steps.append(alpha)
    alpha_all = torch.stack(alpha_steps, dim=1)

    batch_idx = torch.arange(batch_size, device=device)
    last_frame = (frame_lens - 1).clamp_min(0)
    last_state = (state_lens - 1).clamp_min(0)
    prev_state = (state_lens - 2).clamp_min(0)
    terminal_alpha = alpha_all[batch_idx, last_frame]
    final_blank_score = terminal_alpha[batch_idx, last_state]
    final_token_score = terminal_alpha[batch_idx, prev_state]
    log_score = torch.where(
        last_state == prev_state,
        final_blank_score,
        torch.logaddexp(final_blank_score, final_token_score),
    )

    can_skip_successor = torch.zeros_like(can_skip)
    if state_count > 2:
        can_skip_successor[:, :-2] = can_skip[:, 2:]

    terminal_state = (state_idx.unsqueeze(0) == last_state.unsqueeze(1)) | (
        state_idx.unsqueeze(0) == prev_state.unsqueeze(1)
    )
    terminal_beta = torch.where(
        terminal_state,
        lp.new_zeros(()),
        lp.new_full((), _NEG),
    )
    beta_after = lp.new_full((batch_size, state_count), _NEG)
    beta_steps = [None] * num_frames
    for t in range(num_frames - 1, -1, -1):
        is_terminal = (last_frame == t).unsqueeze(1)
        if t == num_frames - 1:
            recurrence = beta_after
        else:
            future = emit[:, t + 1] + beta_after
            next1 = torch.cat([future[:, 1:], neg1], dim=1)
            next2 = torch.cat([future[:, 2:], neg2], dim=1)
            recurrence = torch.logaddexp(future, next1)
            recurrence = torch.where(
                can_skip_successor,
                torch.logaddexp(recurrence, next2),
                recurrence,
            )
        beta = torch.where(
            is_terminal,
            terminal_beta,
            torch.where(
                (t < last_frame).unsqueeze(1),
                recurrence,
                recurrence.new_full((), _NEG),
            ),
        )
        beta_steps[t] = beta
        beta_after = beta
    beta_all = torch.stack(beta_steps, dim=1)

    frame_mask = (
        torch.arange(num_frames, device=device).unsqueeze(0)
        < frame_lens.unsqueeze(1)
    )
    valid_state = frame_mask.unsqueeze(2) & state_mask.unsqueeze(1)
    log_occupancy = alpha_all + beta_all - log_score[:, None, None]
    log_occupancy = log_occupancy.masked_fill(~valid_state, _NEG)
    occupancy = torch.exp(log_occupancy).nan_to_num(
        0.0,
        posinf=0.0,
        neginf=0.0,
    )

    token_indices = 2 * torch.arange(num_tokens, device=device) + 1
    token_occupancy = occupancy.index_select(2, token_indices)
    token_mask = (
        torch.arange(num_tokens, device=device).unsqueeze(0)
        < label_lens.unsqueeze(1)
    )
    return token_occupancy.masked_fill(
        ~(frame_mask.unsqueeze(2) & token_mask.unsqueeze(1)),
        0.0,
    )


def temporal_plan_w1_loss(
    student_occupancy: Tensor,
    teacher_plan: Tensor,
    frame_lens: Tensor,
    label_lens: Tensor,
) -> Tensor:
    """Mean per-token temporal W1 from CTC student to fixed plan teacher.

    The scalar distance is symmetric, but the gradient direction is not:
    ``student_occupancy`` remains differentiable and ``teacher_plan`` is
    detached defensively. A one-frame displacement costs one encoder frame.
    """
    if teacher_plan.shape != student_occupancy.shape:
        raise ValueError(
            "teacher_plan and student_occupancy must have identical shapes: "
            f"{teacher_plan.shape} != {student_occupancy.shape}"
        )

    tiny = 1.0e-8
    _, num_frames, num_tokens = student_occupancy.shape
    device = student_occupancy.device
    frame_mask = (
        torch.arange(num_frames, device=device).unsqueeze(0)
        < frame_lens.unsqueeze(1)
    )
    token_mask = (
        torch.arange(num_tokens, device=device).unsqueeze(0)
        < label_lens.unsqueeze(1)
    )
    valid = frame_mask.unsqueeze(2) & token_mask.unsqueeze(1)

    student = student_occupancy.float().masked_fill(~valid, 0.0)
    teacher = teacher_plan.detach().float().masked_fill(~valid, 0.0)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(tiny)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(tiny)

    cdf_error = (
        student.cumsum(dim=1) - teacher.cumsum(dim=1)
    ).abs().masked_fill(~valid, 0.0)
    denom = label_lens.float().clamp_min(1.0)
    return cdf_error.sum(dim=(1, 2)) / denom


def plan_w1_loss_for_gates(
    teacher_plan: Tensor,
    effective_log_probs: Tensor,
    prior_log_probs: Tensor,
    posterior_log_probs: Tensor,
    labels: Tensor,
    frame_lens: Tensor,
    label_lens: Tensor,
    dual_gate: bool,
    prior_weight: float,
    posterior_weight: float,
    blank_id: int = 0,
    precomputed_effective_occupancy: Tensor | None = None,
    precomputed_prior_occupancy: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Match the plan to the CTC occupancy used by each supervised gate.

    In dual-gate mode the X-only prior must receive the geometry loss directly;
    constraining only the posterior/effective gate would make primary
    transcript-independent evaluation depend on indirect KL transfer.

    Returns ``(combined, prior, posterior)`` per-utterance loss vectors.  In
    non-dual mode, ``combined`` is computed from ``effective_log_probs`` and the
    two branch diagnostics are zero.
    """
    zero = effective_log_probs.new_zeros(effective_log_probs.size(0))
    if not dual_gate:
        occupancy = precomputed_effective_occupancy
        if occupancy is None:
            occupancy = ctc_token_occupancy_batched(
                log_probs=effective_log_probs,
                labels=labels,
                frame_lens=frame_lens,
                label_lens=label_lens,
                blank_id=blank_id,
            )
        combined = temporal_plan_w1_loss(
            student_occupancy=occupancy,
            teacher_plan=teacher_plan,
            frame_lens=frame_lens,
            label_lens=label_lens,
        )
        return combined, zero, zero

    prior_occupancy = precomputed_prior_occupancy
    if prior_occupancy is None:
        prior_occupancy = ctc_token_occupancy_batched(
            log_probs=prior_log_probs,
            labels=labels,
            frame_lens=frame_lens,
            label_lens=label_lens,
            blank_id=blank_id,
        )
    prior_loss = temporal_plan_w1_loss(
        student_occupancy=prior_occupancy,
        teacher_plan=teacher_plan,
        frame_lens=frame_lens,
        label_lens=label_lens,
    )
    if posterior_weight > 0.0:
        posterior_occupancy = ctc_token_occupancy_batched(
            log_probs=posterior_log_probs,
            labels=labels,
            frame_lens=frame_lens,
            label_lens=label_lens,
            blank_id=blank_id,
        )
        posterior_loss = temporal_plan_w1_loss(
            student_occupancy=posterior_occupancy,
            teacher_plan=teacher_plan,
            frame_lens=frame_lens,
            label_lens=label_lens,
        )
    else:
        posterior_loss = zero
    combined = (
        float(prior_weight) * prior_loss
        + float(posterior_weight) * posterior_loss
    )
    return combined, prior_loss, posterior_loss
