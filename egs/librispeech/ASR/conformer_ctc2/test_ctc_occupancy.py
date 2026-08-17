"""Tests for ctc_occupancy.py.

Run as a script (no pytest in env):
    python test_ctc_occupancy.py
"""

import itertools

import torch
import torch.nn.functional as F

from ctc_occupancy import ctc_fb_occupancy, fb_posterior_consistency_loss


def _brute_force_occupancy(log_probs, targets):
    """Enumerate all CTC paths for one short utterance.

    log_probs: [T, V] (blank=0), targets: [U].  Returns (occ [T, U], log_Z).
    """
    T, V = log_probs.shape
    y = targets.tolist()
    U = len(y)

    def collapse(path):
        out = []
        prev = None
        for p in path:
            if p != prev and p != 0:
                out.append(p)
            prev = p
        return out

    occ = torch.zeros(T, U, dtype=torch.float64)
    log_terms = []
    masses = []
    for path in itertools.product(range(V), repeat=T):
        if collapse(path) != y:
            continue
        lp = sum(log_probs[t, k].item() for t, k in enumerate(path))
        log_terms.append(lp)
        # Map each non-blank frame to its target position.
        pos = -1
        prev = None
        assign = []
        for t, k in enumerate(path):
            if k != 0 and k != prev:
                pos += 1
            assign.append(pos if k != 0 else None)
            prev = k
        masses.append((lp, assign))
    log_z = torch.logsumexp(torch.tensor(log_terms, dtype=torch.float64), dim=0)
    for lp, assign in masses:
        w = torch.exp(torch.tensor(lp, dtype=torch.float64) - log_z)
        for t, u in enumerate(assign):
            if u is not None:
                occ[t, u] += w
    return occ, log_z


def test_logz_matches_torch_ctc():
    torch.manual_seed(0)
    B, T, V, U = 5, 17, 7, 5
    lp = torch.log_softmax(torch.randn(B, T, V), dim=-1)
    targets = torch.randint(1, V, (B, U))
    ilens = torch.tensor([17, 15, 12, 9, 6])
    tlens = torch.tensor([5, 4, 3, 2, 1])

    _, log_z = ctc_fb_occupancy(lp, targets, ilens, tlens)

    flat = torch.cat([targets[b, : tlens[b]] for b in range(B)])
    ref = F.ctc_loss(
        lp.transpose(0, 1), flat, ilens, tlens, blank=0, reduction="none"
    )
    assert torch.allclose(-log_z, ref, atol=1e-4), (log_z, -ref)
    print("test_logz_matches_torch_ctc OK")


def test_occupancy_matches_brute_force():
    torch.manual_seed(1)
    for T, V, y in [(4, 3, [1, 2]), (5, 4, [2, 2]), (6, 3, [1, 2, 1])]:
        lp = torch.log_softmax(torch.randn(T, V), dim=-1)
        targets = torch.tensor(y)
        occ_ref, log_z_ref = _brute_force_occupancy(lp.double(), targets)
        occ, log_z = ctc_fb_occupancy(
            lp.unsqueeze(0),
            targets.unsqueeze(0),
            torch.tensor([T]),
            torch.tensor([len(y)]),
        )
        assert torch.allclose(log_z.double(), log_z_ref, atol=1e-5)
        assert torch.allclose(occ[0].double(), occ_ref, atol=1e-5), (
            occ[0],
            occ_ref,
        )
    print("test_occupancy_matches_brute_force OK")


def test_batched_matches_per_utterance():
    torch.manual_seed(2)
    B, T, V, U = 6, 23, 9, 7
    lp = torch.log_softmax(torch.randn(B, T, V), dim=-1)
    targets = torch.randint(1, V, (B, U))
    ilens = torch.tensor([23, 20, 18, 11, 7, 5])
    tlens = torch.tensor([7, 6, 4, 3, 2, 1])
    occ, log_z = ctc_fb_occupancy(lp, targets, ilens, tlens)
    for b in range(B):
        ob, zb = ctc_fb_occupancy(
            lp[b : b + 1, : ilens[b]],
            targets[b : b + 1, : tlens[b]],
            ilens[b : b + 1],
            tlens[b : b + 1],
        )
        assert torch.allclose(zb, log_z[b], atol=1e-4)
        assert torch.allclose(ob[0], occ[b, : ilens[b], : tlens[b]], atol=1e-5)
    # Padded regions carry no occupancy.
    for b in range(B):
        assert occ[b, ilens[b] :].abs().sum() == 0
        assert occ[b, :, tlens[b] :].abs().sum() == 0
    print("test_batched_matches_per_utterance OK")


def test_frame_mass_conservation():
    torch.manual_seed(3)
    B, T, V, U = 3, 19, 8, 6
    lp = torch.log_softmax(torch.randn(B, T, V), dim=-1)
    targets = torch.randint(1, V, (B, U))
    ilens = torch.tensor([19, 14, 10])
    tlens = torch.tensor([6, 4, 2])
    occ, _ = ctc_fb_occupancy(lp, targets, ilens, tlens)
    # Per valid frame, token occupancy + blank occupancy = 1, so the token
    # part must lie in [0, 1].
    per_frame = occ.sum(dim=2)
    for b in range(B):
        v = per_frame[b, : ilens[b]]
        assert (v <= 1.0 + 1e-4).all() and (v >= -1e-6).all()
    print("test_frame_mass_conservation OK")


def test_loss_gradient_flows_and_is_finite():
    torch.manual_seed(4)
    B, T, V, U = 4, 21, 9, 6
    logits = torch.randn(B, T, V, requires_grad=True)
    lp = torch.log_softmax(logits, dim=-1)
    targets = torch.randint(1, V, (B, U))
    ilens = torch.tensor([21, 17, 12, 8])
    tlens = torch.tensor([6, 5, 3, 1])
    plan = torch.rand(B, T, U)
    plan = plan / plan.sum(dim=(1, 2), keepdim=True)
    loss = fb_posterior_consistency_loss(plan, lp, targets, ilens, tlens)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    # The plan is a detached target: no gradient w.r.t. the plan input.
    print("test_loss_gradient_flows_and_is_finite OK")


if __name__ == "__main__":
    test_logz_matches_torch_ctc()
    test_occupancy_matches_brute_force()
    test_batched_matches_per_utterance()
    test_frame_mass_conservation()
    test_loss_gradient_flows_and_is_finite()
    print("All tests passed.")
