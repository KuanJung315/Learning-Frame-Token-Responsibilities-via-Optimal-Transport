"""Exact conditional CTC path entropy."""

from typing import Tuple

import torch


def _merge_path_sets(
    log_masses: torch.Tensor,
    entropies: torch.Tensor,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge disjoint weighted path sets in the entropy semiring.

    This computes conditional entropy directly instead of subtracting two
    large negative values (``log Z - E[log w]``), which is inaccurate for
    long CTC sequences in float32.
    """
    reachable = torch.isfinite(log_masses)
    has_path = reachable.any(dim=dim, keepdim=True)

    # Avoid undefined gradients from logsumexp([-inf, ...]) while preserving
    # the real -inf entries whenever at least one component is reachable.
    safe_for_logsumexp = torch.where(
        has_path, log_masses, torch.zeros_like(log_masses)
    )
    merged_log_mass = torch.logsumexp(safe_for_logsumexp, dim=dim)
    merged_log_mass = torch.where(
        has_path.squeeze(dim),
        merged_log_mass,
        log_masses.new_full(merged_log_mass.shape, float("-inf")),
    )

    safe_log_mix = torch.where(
        reachable,
        log_masses - merged_log_mass.unsqueeze(dim),
        torch.zeros_like(log_masses),
    )
    mix = torch.where(reachable, safe_log_mix.exp(), torch.zeros_like(log_masses))
    merged_entropy = (mix * (entropies - safe_log_mix)).sum(dim=dim)
    merged_entropy = torch.where(
        has_path.squeeze(dim), merged_entropy, torch.zeros_like(merged_entropy)
    )
    return merged_log_mass, merged_entropy


def _prepare_targets(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    batch_size: int,
    num_classes: int,
    blank: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return padded active labels and their mask."""
    max_target_length = int(target_lengths.max().item())
    positions = torch.arange(max_target_length, device=device)
    target_mask = positions.unsqueeze(0) < target_lengths.unsqueeze(1)

    targets = torch.as_tensor(targets, device=device, dtype=torch.long)
    if targets.ndim == 2:
        if targets.shape[0] != batch_size:
            raise ValueError(
                f"targets has batch size {targets.shape[0]}, expected {batch_size}"
            )
        if targets.shape[1] < max_target_length:
            raise ValueError("targets is shorter than a value in target_lengths")
        labels = targets[:, :max_target_length]
    elif targets.ndim == 1:
        expected = int(target_lengths.sum().item())
        if targets.numel() != expected:
            raise ValueError(
                f"1-D targets has {targets.numel()} elements, expected {expected}"
            )
        if max_target_length == 0:
            labels = targets.new_empty((batch_size, 0))
        else:
            offsets = target_lengths.cumsum(0) - target_lengths
            flat_indices = offsets.unsqueeze(1) + positions.unsqueeze(0)
            flat_indices = flat_indices.clamp(max=max(targets.numel() - 1, 0))
            labels = targets[flat_indices]
    else:
        raise ValueError("targets must be a padded 2-D tensor or concatenated 1-D tensor")

    if target_mask.any():
        active_labels = labels[target_mask]
        if ((active_labels < 0) | (active_labels >= num_classes)).any():
            raise ValueError("active target labels must be valid class indices")
        if (active_labels == blank).any():
            raise ValueError("active target labels must not contain blank")

    labels = torch.where(target_mask, labels, torch.full_like(labels, blank))
    return labels, target_mask


def ctc_path_entropy(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact CTC log-likelihood and conditional path entropy.

    Args:
      log_probs:
        Log probabilities (or log weights) shaped ``(T, B, C)``.
      input_lengths:
        Number of active frames for each utterance, shaped ``(B,)``.
      targets:
        Padded targets shaped ``(B, S)`` or concatenated targets shaped
        ``(sum(target_lengths),)``.
      target_lengths:
        Number of active target labels for each utterance, shaped ``(B,)``.
      blank:
        Blank class index.

    Returns:
      A pair ``(log_likelihood, entropy)``, each shaped ``(B,)``. The entropy
      is the exact conditional entropy over complete CTC paths:

      It is evaluated directly with an entropy-semiring forward pass, avoiding
      cancellation from ``log Z - E[log w]`` on long sequences.

      For an impossible transcript, log-likelihood is ``-inf`` and entropy is
      returned as zero because the conditional path distribution is undefined.
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape (T, B, C)")
    if not log_probs.is_floating_point():
        raise TypeError("log_probs must be floating point")

    num_frames, batch_size, num_classes = log_probs.shape
    if num_frames == 0:
        raise ValueError("log_probs must contain at least one frame")
    if not 0 <= blank < num_classes:
        raise ValueError(f"blank must be in [0, {num_classes})")

    device = log_probs.device
    input_lengths = torch.as_tensor(input_lengths, device=device, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, device=device, dtype=torch.long)
    if input_lengths.shape != (batch_size,):
        raise ValueError(f"input_lengths must have shape ({batch_size},)")
    if target_lengths.shape != (batch_size,):
        raise ValueError(f"target_lengths must have shape ({batch_size},)")
    if ((input_lengths < 1) | (input_lengths > num_frames)).any():
        raise ValueError(f"input_lengths must be in [1, {num_frames}]")
    if (target_lengths < 0).any():
        raise ValueError("target_lengths must be non-negative")

    labels, _ = _prepare_targets(
        targets,
        target_lengths,
        batch_size,
        num_classes,
        blank,
        device,
    )

    max_target_length = labels.shape[1]
    max_states = 2 * max_target_length + 1
    state_positions = torch.arange(max_states, device=device)
    state_lengths = 2 * target_lengths + 1
    state_mask = state_positions.unsqueeze(0) < state_lengths.unsqueeze(1)

    extended_targets = torch.full(
        (batch_size, max_states), blank, dtype=torch.long, device=device
    )
    if max_target_length:
        extended_targets[:, 1::2] = labels

    can_skip = torch.zeros(
        (batch_size, max_states), dtype=torch.bool, device=device
    )
    if max_states > 2:
        can_skip[:, 2:] = (
            (extended_targets[:, 2:] != blank)
            & (extended_targets[:, 2:] != extended_targets[:, :-2])
        )
    can_skip &= state_mask

    emissions = log_probs.gather(
        2, extended_targets.unsqueeze(0).expand(num_frames, -1, -1)
    )
    neg_inf = log_probs.new_tensor(float("-inf"))
    neg_column = log_probs.new_full((batch_size, 1), float("-inf"))
    neg_two_columns = log_probs.new_full((batch_size, 2), float("-inf"))

    start_mask = state_positions.unsqueeze(0) == 0
    if max_states > 1:
        start_mask = start_mask | (
            (state_positions.unsqueeze(0) == 1)
            & (target_lengths.unsqueeze(1) > 0)
        )
    alpha_log_mass = torch.where(start_mask, emissions[0], neg_inf)
    alpha_entropy = torch.zeros_like(alpha_log_mass)

    for t in range(1, num_frames):
        from_previous_log = torch.cat(
            [neg_column, alpha_log_mass[:, :-1]], dim=1
        )
        from_previous_entropy = torch.cat(
            [torch.zeros_like(neg_column), alpha_entropy[:, :-1]], dim=1
        )
        if max_states == 1:
            from_two_back_log = neg_column
            from_two_back_entropy = torch.zeros_like(neg_column)
        else:
            from_two_back_log = torch.cat(
                [neg_two_columns, alpha_log_mass[:, :-2]], dim=1
            )
            from_two_back_entropy = torch.cat(
                [torch.zeros_like(neg_two_columns), alpha_entropy[:, :-2]], dim=1
            )
        from_two_back_log = torch.where(can_skip, from_two_back_log, neg_inf)
        from_two_back_entropy = torch.where(
            can_skip, from_two_back_entropy, torch.zeros_like(from_two_back_entropy)
        )
        predecessor_log, predecessor_entropy = _merge_path_sets(
            torch.stack(
                [alpha_log_mass, from_previous_log, from_two_back_log], dim=0
            ),
            torch.stack(
                [
                    alpha_entropy,
                    from_previous_entropy,
                    from_two_back_entropy,
                ],
                dim=0,
            ),
            dim=0,
        )
        candidate_log_mass = predecessor_log + emissions[t]
        active = (t < input_lengths).unsqueeze(1) & state_mask
        # Retaining completed utterances avoids storing every alpha timestep.
        alpha_log_mass = torch.where(active, candidate_log_mass, alpha_log_mass)
        alpha_entropy = torch.where(active, predecessor_entropy, alpha_entropy)

    batch_indices = torch.arange(batch_size, device=device)
    last_states = state_lengths - 1
    previous_states = (last_states - 1).clamp(min=0)
    final_last_log = alpha_log_mass[batch_indices, last_states]
    final_last_entropy = alpha_entropy[batch_indices, last_states]
    final_previous_log = torch.where(
        target_lengths > 0,
        alpha_log_mass[batch_indices, previous_states],
        neg_inf,
    )
    final_previous_entropy = torch.where(
        target_lengths > 0,
        alpha_entropy[batch_indices, previous_states],
        torch.zeros_like(final_last_entropy),
    )
    log_likelihood, entropy = _merge_path_sets(
        torch.stack([final_last_log, final_previous_log], dim=0),
        torch.stack([final_last_entropy, final_previous_entropy], dim=0),
        dim=0,
    )
    return log_likelihood, entropy


__all__ = ["ctc_path_entropy"]
