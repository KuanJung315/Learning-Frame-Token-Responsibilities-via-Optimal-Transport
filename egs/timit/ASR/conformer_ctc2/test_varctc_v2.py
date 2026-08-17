import torch
import torch.nn.functional as F

from blank_gate_v2 import BlankGateHeadV2
from ctc_plan_consistency import ctc_token_occupancy_batched, temporal_plan_w1_loss
from ot_fgw import vi_fgw_loss_v2_batched
from ot_prior_v2 import vi_ot_loss_v2, vi_ot_loss_v2_batched
from varctc_v2_utils import build_gated_log_probs_v2


def test_gated_log_probs_normalized():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 9)
    log_p_nonblank = F.log_softmax(logits, dim=-1)
    alpha = torch.rand(2, 5)

    log_p_gated = build_gated_log_probs_v2(log_p_nonblank, alpha)
    probs = log_p_gated.exp().sum(dim=-1)

    assert log_p_gated.shape == (2, 5, 10)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1.0e-5)


def test_cross_attention_gate_masks_padding_frames():
    torch.manual_seed(1)
    gate = BlankGateHeadV2(d_model=4, vocab_size=8, d_attn=6)
    encoder_out = torch.randn(2, 6, 4)
    targets = torch.tensor([[1, 2, 3], [4, 5, 0]])
    target_lengths = torch.tensor([3, 2])
    encoder_out_lens = torch.tensor([6, 4])

    alpha = gate(encoder_out, targets, target_lengths, encoder_out_lens)

    assert alpha.shape == (2, 6)
    assert torch.isfinite(alpha).all()
    assert torch.all(alpha[1, 4:] == 0)
    assert alpha[0].std(unbiased=False) > 0


def test_vi_ot_loss_v2_gradient_is_decoupled_from_gate():
    torch.manual_seed(2)
    logits = torch.randn(5, 7, requires_grad=True)
    log_p_nonblank = F.log_softmax(logits, dim=-1)
    alpha = torch.rand(5, requires_grad=True)
    labels = torch.tensor([1, 3, 7], dtype=torch.long)
    bpe_lengths = torch.tensor([1.0, 2.0, 1.0])

    loss = vi_ot_loss_v2(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        bpe_lengths=bpe_lengths,
        column_marginal_type="bpe",
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    assert alpha.grad is None


def test_vi_ot_loss_v2_acoustic_column_marginal():
    torch.manual_seed(3)
    logits = torch.randn(6, 8, requires_grad=True)
    log_p_nonblank = F.log_softmax(logits, dim=-1)
    alpha = torch.rand(6, requires_grad=True)
    labels = torch.tensor([2, 4, 6], dtype=torch.long)

    loss = vi_ot_loss_v2(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        bpe_lengths=None,
        column_marginal_type="acoustic",
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    assert alpha.grad is None


def test_acoustic_column_marginal_is_independent_of_positional_bias():
    torch.manual_seed(5)
    log_p_nonblank = F.log_softmax(torch.randn(19, 11), dim=-1)
    alpha = torch.rand(19)
    labels = torch.tensor([1, 4, 7, 9], dtype=torch.long)

    _, plan_no_pos = vi_ot_loss_v2(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        column_marginal_type="acoustic",
        beta_pos=0.0,
        eps=0.1,
        iters=80,
        return_plan=True,
    )
    _, plan_strong_pos = vi_ot_loss_v2(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        column_marginal_type="acoustic",
        beta_pos=50.0,
        eps=0.1,
        iters=80,
        return_plan=True,
    )

    assert plan_no_pos is not None and plan_strong_pos is not None
    assert torch.allclose(
        plan_no_pos.sum(dim=0),
        plan_strong_pos.sum(dim=0),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def _ot_kwargs(col_type):
    return dict(
        column_marginal_type=col_type,
        alpha_smooth_mix=0.1,
        bpe_col_floor=0.05,
        token_prior_sigma=0.15,
        token_prior_score_temp=1.0,
        token_prior_floor=0.05,
        eps=0.3,
        iters=10,
        beta_pos=1.0,
    )


def _run_batched_equivalence(col_type):
    """Batched OT == stacked per-utterance OT (loss values and gradients)."""
    torch.manual_seed(7)
    Vm1 = 12
    frame_lens = [17, 9, 23, 1, 14]   # include a length-1 utterance (linspace edge)
    label_lens = [4, 3, 6, 2, 1]      # include a length-1 label sequence
    B = len(frame_lens)
    T_max, U_max = max(frame_lens), max(label_lens)

    logits = torch.randn(B, T_max, Vm1, dtype=torch.float64)
    log_p = F.log_softmax(logits, dim=-1)
    alpha = torch.rand(B, T_max, dtype=torch.float64)
    labels = torch.randint(1, Vm1 + 1, (B, U_max))
    flen = torch.tensor(frame_lens)
    llen = torch.tensor(label_lens)
    bpe = torch.randint(1, 4, (B, U_max)).double() if col_type == "bpe" else None

    # ── per-utterance reference (clone leaves so grads are independent) ──
    log_p_loop = log_p.clone().requires_grad_(True)
    ref_losses = []
    for i in range(B):
        L, Uq = frame_lens[i], label_lens[i]
        loss_i = vi_ot_loss_v2(
            log_p_nonblank=log_p_loop[i, :L],
            alpha=alpha[i, :L],
            labels=labels[i, :Uq],
            bpe_lengths=(bpe[i, :Uq] if bpe is not None else None),
            **_ot_kwargs(col_type),
        )
        ref_losses.append(loss_i)
    ref = torch.stack(ref_losses)
    ref.sum().backward()
    ref_grad = log_p_loop.grad.clone()

    # ── batched ──
    log_p_b = log_p.clone().requires_grad_(True)
    got = vi_ot_loss_v2_batched(
        log_p_nonblank=log_p_b,
        alpha=alpha,
        labels=labels,
        frame_lens=flen,
        label_lens=llen,
        bpe_lengths=bpe,
        **_ot_kwargs(col_type),
    )
    got.sum().backward()

    # Both paths cast to float32 internally (.float()), so equivalence is at
    # float32 precision; logsumexp reduction order differs ⇒ ~1e-5 differences.
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (
        f"[{col_type}] loss mismatch (max abs diff "
        f"{(got - ref).abs().max().item():.2e}):\n ref={ref}\n got={got}"
    )
    # gradient only on valid frames must match; padded frames stay zero
    assert torch.allclose(log_p_b.grad, ref_grad, atol=1e-4, rtol=1e-4), (
        f"[{col_type}] grad mismatch (max abs diff "
        f"{(log_p_b.grad - ref_grad).abs().max().item():.2e})"
    )


def test_vi_ot_loss_v2_batched_matches_loop_acoustic():
    _run_batched_equivalence("acoustic")


def test_vi_ot_loss_v2_batched_matches_loop_bpe():
    _run_batched_equivalence("bpe")


def test_vi_ot_loss_v2_batched_matches_loop_uniform():
    _run_batched_equivalence("uniform")


def test_vi_ot_loss_v2_batched_gradient_is_decoupled_from_gate():
    torch.manual_seed(11)
    B, T, U, Vm1 = 3, 7, 4, 9
    logits = torch.randn(B, T, Vm1, requires_grad=True)
    log_p_nonblank = F.log_softmax(logits, dim=-1)
    alpha = torch.rand(B, T, requires_grad=True)
    labels = torch.randint(1, Vm1 + 1, (B, U))
    frame_lens = torch.tensor([7, 5, 3])
    label_lens = torch.tensor([4, 3, 2])

    loss = vi_ot_loss_v2_batched(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        frame_lens=frame_lens,
        label_lens=label_lens,
        column_marginal_type="acoustic",
    )
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    assert alpha.grad is None


def test_ctc_token_occupancy_is_monotonic_and_differentiable():
    torch.manual_seed(13)
    B, T, V, U = 2, 12, 8, 4
    logits = torch.randn(B, T, V, requires_grad=True)
    log_probs = F.log_softmax(logits, dim=-1)
    labels = torch.tensor([[1, 2, 2, 3], [4, 2, 5, 0]])
    frame_lens = torch.tensor([12, 9])
    label_lens = torch.tensor([4, 3])

    occupancy = ctc_token_occupancy_batched(
        log_probs, labels, frame_lens, label_lens
    )
    assert occupancy.shape == (B, T, U)
    assert occupancy.requires_grad
    assert torch.isfinite(occupancy).all()

    frame = torch.arange(T, dtype=occupancy.dtype)
    for b, num_labels in enumerate(label_lens.tolist()):
        values = occupancy[b, :, :num_labels]
        centers = (values * frame[:, None]).sum(dim=0) / values.sum(dim=0).clamp_min(
            1.0e-8
        )
        assert torch.all(centers[1:] >= centers[:-1] - 1.0e-5)

    teacher = torch.rand_like(occupancy)
    loss = temporal_plan_w1_loss(
        student_occupancy=occupancy,
        teacher_plan=teacher,
        frame_lens=frame_lens,
        label_lens=label_lens,
    ).sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_temporal_plan_w1_zero_target_and_ctc_student_gradient():
    torch.manual_seed(17)
    B, T, U = 2, 9, 4
    target = torch.rand(B, T, U)
    frame_lens = torch.tensor([9, 6])
    label_lens = torch.tensor([4, 3])

    zero = temporal_plan_w1_loss(
        student_occupancy=target,
        teacher_plan=target,
        frame_lens=frame_lens,
        label_lens=label_lens,
    )
    assert torch.allclose(zero, torch.zeros_like(zero), atol=1.0e-6)

    student = torch.rand(B, T, U, requires_grad=True)
    teacher = torch.rand(B, T, U, requires_grad=True)
    loss = temporal_plan_w1_loss(
        student_occupancy=student,
        teacher_plan=teacher,
        frame_lens=frame_lens,
        label_lens=label_lens,
    ).sum()
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_temporal_plan_w1_is_measured_in_frames():
    target = torch.zeros(1, 5, 1)
    target[0, 1, 0] = 1.0
    plan = torch.zeros_like(target)
    plan[0, 2, 0] = 1.0

    loss = temporal_plan_w1_loss(
        student_occupancy=plan,
        teacher_plan=target,
        frame_lens=torch.tensor([5]),
        label_lens=torch.tensor([1]),
    )
    assert torch.allclose(loss, torch.ones_like(loss), atol=1.0e-6)


def test_fgw_plan_teaches_ctc_occupancy_gradient():
    torch.manual_seed(23)
    B, T, U, Vm1 = 2, 12, 4, 7
    logits = torch.randn(B, T, Vm1, requires_grad=True)
    log_p_nonblank = F.log_softmax(logits, dim=-1)
    gate_logits = torch.randn(B, T, requires_grad=True)
    alpha = torch.sigmoid(gate_logits)
    labels = torch.tensor([[1, 2, 2, 3], [4, 2, 5, 0]])
    frame_lens = torch.tensor([12, 9])
    label_lens = torch.tensor([4, 3])
    acoustic_features = torch.randn(B, T, 8)

    _, teacher_plan = vi_fgw_loss_v2_batched(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        frame_lens=frame_lens,
        label_lens=label_lens,
        acoustic_features=acoustic_features,
        column_marginal_type="acoustic",
        eps=0.2,
        iters=8,
        beta_pos=2.0,
        lambda_gw=0.2,
        n_outer=2,
        return_plan=True,
    )
    assert not teacher_plan.requires_grad

    ctc_log_probs = build_gated_log_probs_v2(log_p_nonblank, alpha)
    student_occupancy = ctc_token_occupancy_batched(
        log_probs=ctc_log_probs,
        labels=labels,
        frame_lens=frame_lens,
        label_lens=label_lens,
        blank_id=0,
    )
    loss = temporal_plan_w1_loss(
        student_occupancy=student_occupancy,
        teacher_plan=teacher_plan,
        frame_lens=frame_lens,
        label_lens=label_lens,
    ).sum()
    loss.backward()

    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert gate_logits.grad is not None and gate_logits.grad.abs().sum() > 0
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(gate_logits.grad).all()


def test_vi_ot_loss_v2_batched_can_return_differentiable_plan():
    torch.manual_seed(19)
    B, T, U, Vm1 = 2, 11, 4, 7
    logits = torch.randn(B, T, Vm1, requires_grad=True)
    log_p = F.log_softmax(logits, dim=-1)
    alpha = torch.rand(B, T, requires_grad=True)
    labels = torch.randint(1, Vm1 + 1, (B, U))
    frame_lens = torch.tensor([11, 8])
    label_lens = torch.tensor([4, 3])

    _, plan = vi_ot_loss_v2_batched(
        log_p_nonblank=log_p,
        alpha=alpha,
        labels=labels,
        frame_lens=frame_lens,
        label_lens=label_lens,
        column_marginal_type="acoustic",
        return_plan=True,
        differentiable_plan=True,
    )
    (plan * torch.randn_like(plan)).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert alpha.grad is not None and alpha.grad.abs().sum() > 0


if __name__ == "__main__":
    test_gated_log_probs_normalized()
    test_cross_attention_gate_masks_padding_frames()
    test_vi_ot_loss_v2_gradient_is_decoupled_from_gate()
    test_vi_ot_loss_v2_acoustic_column_marginal()
    test_acoustic_column_marginal_is_independent_of_positional_bias()
    test_vi_ot_loss_v2_batched_matches_loop_acoustic()
    test_vi_ot_loss_v2_batched_matches_loop_bpe()
    test_vi_ot_loss_v2_batched_matches_loop_uniform()
    test_vi_ot_loss_v2_batched_gradient_is_decoupled_from_gate()
    test_ctc_token_occupancy_is_monotonic_and_differentiable()
    test_temporal_plan_w1_zero_target_and_ctc_student_gradient()
    test_temporal_plan_w1_is_measured_in_frames()
    test_fgw_plan_teaches_ctc_occupancy_gradient()
    test_vi_ot_loss_v2_batched_can_return_differentiable_plan()
    print("All tests passed.")
