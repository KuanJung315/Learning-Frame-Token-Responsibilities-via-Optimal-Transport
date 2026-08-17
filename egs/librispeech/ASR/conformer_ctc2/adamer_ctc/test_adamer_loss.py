import itertools

import torch
import torch.nn.functional as F

from loss import ctc_path_entropy


def _collapse(path, blank=0):
    deduplicated = [label for label, _ in itertools.groupby(path)]
    return tuple(label for label in deduplicated if label != blank)


def _brute_force(log_probs, target, blank=0):
    num_frames, num_classes = log_probs.shape
    path_log_weights = []
    for path in itertools.product(range(num_classes), repeat=num_frames):
        if _collapse(path, blank) == tuple(target):
            frame_indices = torch.arange(num_frames)
            path_indices = torch.tensor(path)
            path_log_weights.append(log_probs[frame_indices, path_indices].sum())

    assert path_log_weights, "test case must have at least one valid CTC path"
    path_log_weights = torch.stack(path_log_weights)
    log_likelihood = torch.logsumexp(path_log_weights, dim=0)
    posterior = (path_log_weights - log_likelihood).exp()
    entropy = -(posterior * (path_log_weights - log_likelihood)).sum()
    return log_likelihood, entropy


def _batch_example(requires_grad=False):
    torch.manual_seed(42)
    logits = torch.randn(5, 3, 4, dtype=torch.float64, requires_grad=requires_grad)
    log_probs = logits.log_softmax(dim=-1)
    input_lengths = torch.tensor([4, 5, 3])
    targets = torch.tensor(
        [
            [1, 1, 3],
            [2, 3, 0],
            [3, 0, 0],
        ]
    )
    target_lengths = torch.tensor([2, 2, 1])
    return logits, log_probs, input_lengths, targets, target_lengths


def test_matches_brute_force_for_batch_repeats_and_padding():
    _, log_probs, input_lengths, targets, target_lengths = _batch_example()

    log_likelihood, entropy = ctc_path_entropy(
        log_probs, input_lengths, targets, target_lengths
    )

    for b in range(log_probs.shape[1]):
        target = targets[b, : target_lengths[b]].tolist()
        expected_log_likelihood, expected_entropy = _brute_force(
            log_probs[: input_lengths[b], b], target
        )
        assert torch.allclose(
            log_likelihood[b], expected_log_likelihood, atol=1.0e-10, rtol=1.0e-10
        )
        assert torch.allclose(
            entropy[b], expected_entropy, atol=1.0e-10, rtol=1.0e-10
        )


def test_log_likelihood_matches_torch_ctc_loss():
    _, log_probs, input_lengths, targets, target_lengths = _batch_example()

    log_likelihood, _ = ctc_path_entropy(
        log_probs, input_lengths, targets, target_lengths
    )
    expected = -F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
    )

    assert torch.allclose(log_likelihood, expected, atol=1.0e-10, rtol=1.0e-10)


def test_padding_frames_and_target_padding_are_ignored():
    _, log_probs, input_lengths, targets, target_lengths = _batch_example()
    changed = log_probs.detach().clone()
    changed[4, 0] = torch.tensor([20.0, -10.0, 7.0, 3.0], dtype=changed.dtype)
    changed[3:, 2] = torch.tensor([[-8.0, 4.0, 2.0, 9.0]], dtype=changed.dtype)
    changed_targets = targets.clone()
    changed_targets[0, 2] = -100
    changed_targets[2, 1:] = torch.tensor([-100, -100])

    actual = ctc_path_entropy(log_probs, input_lengths, targets, target_lengths)
    changed_result = ctc_path_entropy(
        changed, input_lengths, changed_targets, target_lengths
    )

    assert torch.allclose(actual[0], changed_result[0], atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(actual[1], changed_result[1], atol=1.0e-10, rtol=1.0e-10)


def test_concatenated_targets_match_padded_targets():
    _, log_probs, input_lengths, targets, target_lengths = _batch_example()
    concatenated = torch.cat(
        [targets[b, : target_lengths[b]] for b in range(targets.shape[0])]
    )

    padded_result = ctc_path_entropy(
        log_probs, input_lengths, targets, target_lengths
    )
    concatenated_result = ctc_path_entropy(
        log_probs, input_lengths, concatenated, target_lengths
    )

    assert torch.allclose(padded_result[0], concatenated_result[0])
    assert torch.allclose(padded_result[1], concatenated_result[1])


def test_empty_targets_have_one_all_blank_path():
    torch.manual_seed(7)
    logits = torch.randn(4, 2, 3, dtype=torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    input_lengths = torch.tensor([4, 2])
    targets = torch.empty((2, 0), dtype=torch.long)
    target_lengths = torch.tensor([0, 0])

    log_likelihood, entropy = ctc_path_entropy(
        log_probs, input_lengths, targets, target_lengths
    )
    expected = torch.stack(
        [
            log_probs[: input_lengths[b], b, 0].sum()
            for b in range(log_probs.shape[1])
        ]
    )

    assert torch.allclose(log_likelihood, expected, atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(entropy, torch.zeros_like(entropy), atol=1.0e-10)


def test_gradients_are_finite():
    logits, log_probs, input_lengths, targets, target_lengths = _batch_example(
        requires_grad=True
    )

    log_likelihood, entropy = ctc_path_entropy(
        log_probs, input_lengths, targets, target_lengths
    )
    objective = (-log_likelihood + 0.3 * entropy).sum()
    objective.backward()

    assert torch.isfinite(log_likelihood).all()
    assert torch.isfinite(entropy).all()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_extreme_logits_remain_finite_and_differentiable():
    logits = (
        torch.tensor(
            [
                [[1000.0, -1000.0, 0.0]],
                [[-1000.0, 1000.0, 0.0]],
                [[1000.0, -1000.0, 0.0]],
            ],
            dtype=torch.float64,
        )
        .clone()
        .requires_grad_()
    )
    log_probs = logits.log_softmax(dim=-1)

    log_likelihood, entropy = ctc_path_entropy(
        log_probs,
        torch.tensor([3]),
        torch.tensor([[1]]),
        torch.tensor([1]),
    )
    (-log_likelihood + entropy).sum().backward()

    assert torch.isfinite(log_likelihood).all()
    assert torch.isfinite(entropy).all()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_long_float32_sequence_has_nonnegative_finite_entropy():
    torch.manual_seed(2026)
    logits = torch.randn(1000, 1, 32, requires_grad=True)
    log_probs = logits.log_softmax(dim=-1)
    targets = torch.randint(1, 32, (1, 20))

    log_likelihood, entropy = ctc_path_entropy(
        log_probs,
        torch.tensor([1000]),
        targets,
        torch.tensor([20]),
    )
    (-log_likelihood + entropy).sum().backward()

    assert torch.isfinite(log_likelihood).all()
    assert torch.isfinite(entropy).all()
    assert (entropy >= 0).all()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


if __name__ == "__main__":
    test_matches_brute_force_for_batch_repeats_and_padding()
    test_log_likelihood_matches_torch_ctc_loss()
    test_padding_frames_and_target_padding_are_ignored()
    test_concatenated_targets_match_padded_targets()
    test_empty_targets_have_one_all_blank_path()
    test_gradients_are_finite()
    test_extreme_logits_remain_finite_and_differentiable()
    test_long_float32_sequence_has_nonnegative_finite_entropy()
