import sys
from pathlib import Path

import torch

ASR_DIR = Path(__file__).resolve().parent.parent
if str(ASR_DIR) not in sys.path:
    sys.path.insert(0, str(ASR_DIR))

from shared_alignment_viz import _compute_ctc_state_occupancy


def _reference_occupancy(log_probs, labels, blank_id=0):
    log_probs = log_probs.to(torch.float64)
    T = log_probs.size(0)
    U = labels.numel()
    ext = labels.new_full((2 * U + 1,), blank_id)
    ext[1::2] = labels
    S = ext.numel()
    emit = log_probs.index_select(1, ext)

    alpha = log_probs.new_full((T, S), float("-inf"))
    alpha[0, 0] = emit[0, 0]
    alpha[0, 1] = emit[0, 1]
    for t in range(1, T):
        for s in range(S):
            prev = [alpha[t - 1, s]]
            if s >= 1:
                prev.append(alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                prev.append(alpha[t - 1, s - 2])
            alpha[t, s] = emit[t, s] + torch.logsumexp(torch.stack(prev), 0)

    beta = log_probs.new_full((T, S), float("-inf"))
    beta[-1, -1] = emit[-1, -1]
    beta[-1, -2] = emit[-1, -2]
    for t in range(T - 2, -1, -1):
        for s in range(S):
            nxt = [beta[t + 1, s]]
            if s + 1 < S:
                nxt.append(beta[t + 1, s + 1])
            if (
                s + 2 < S
                and ext[s + 2] != blank_id
                and ext[s + 2] != ext[s]
            ):
                nxt.append(beta[t + 1, s + 2])
            beta[t, s] = emit[t, s] + torch.logsumexp(torch.stack(nxt), 0)

    log_z = torch.logsumexp(alpha[-1, -2:], 0)
    return torch.exp(alpha + beta - emit - log_z)[:, 1::2].float()


def test_vectorized_ctc_occupancy_matches_reference():
    torch.manual_seed(0)
    for labels in (torch.tensor([1, 2, 3]), torch.tensor([1, 1, 2])):
        log_probs = torch.randn(12, 5).log_softmax(dim=-1)
        expected = _reference_occupancy(log_probs, labels)
        actual = _compute_ctc_state_occupancy(log_probs, labels, blank_id=0)
        torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)
