import torch
import torch.nn.functional as F

from blank_gate_v2 import BlankGateHeadV2
from ctc_plan_consistency import plan_w1_loss_for_gates
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


def test_dual_gate_supervision_trains_both_gates():
    torch.manual_seed(12)
    logits = torch.randn(1, 5, 6)
    log_p_nonblank = F.log_softmax(logits, dim=-1)
    prior_logits = torch.randn(1, 5, requires_grad=True)
    post_logits = torch.randn(1, 5, requires_grad=True)

    prior_log_probs = build_gated_log_probs_v2(
        log_p_nonblank, torch.sigmoid(prior_logits)
    )
    post_log_probs = build_gated_log_probs_v2(
        log_p_nonblank, torch.sigmoid(post_logits)
    )
    frame_targets = torch.tensor([[0, 2, 0, 4, 0]])
    prior_loss = -prior_log_probs.gather(-1, frame_targets.unsqueeze(-1)).mean()
    post_loss = -post_log_probs.gather(-1, frame_targets.unsqueeze(-1)).mean()
    (0.5 * prior_loss + 0.5 * post_loss).backward()

    assert prior_logits.grad is not None
    assert prior_logits.grad.abs().sum() > 0
    assert post_logits.grad is not None
    assert post_logits.grad.abs().sum() > 0


def test_blank_kl_distills_posterior_to_prior_only():
    torch.manual_seed(13)
    post_logits = torch.randn(7, requires_grad=True)
    prior_logits = torch.randn(7, requires_grad=True)
    post = torch.sigmoid(post_logits)
    prior = torch.sigmoid(prior_logits)
    p = post.detach().clamp(1.0e-5, 1.0 - 1.0e-5)
    q = prior.clamp(1.0e-5, 1.0 - 1.0e-5)
    kl = p * (p.log() - q.log()) + (1.0 - p) * (
        (1.0 - p).log() - (1.0 - q).log()
    )
    kl.mean().backward()

    assert post_logits.grad is None
    assert prior_logits.grad is not None
    assert prior_logits.grad.abs().sum() > 0


def test_dual_gate_plan_w1_trains_prior_and_posterior_not_plan():
    torch.manual_seed(14)
    batch_size, num_frames, vocab_size = 1, 7, 4
    nonblank_logits = torch.randn(batch_size, num_frames, vocab_size - 1)
    log_p_nonblank = F.log_softmax(nonblank_logits, dim=-1)
    prior_logits = torch.randn(batch_size, num_frames, requires_grad=True)
    posterior_logits = torch.randn(batch_size, num_frames, requires_grad=True)
    prior_log_probs = build_gated_log_probs_v2(
        log_p_nonblank, torch.sigmoid(prior_logits)
    )
    posterior_log_probs = build_gated_log_probs_v2(
        log_p_nonblank, torch.sigmoid(posterior_logits)
    )
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    teacher_plan = torch.rand(
        batch_size, num_frames, labels.size(1), requires_grad=True
    )
    frame_lens = torch.tensor([num_frames])
    label_lens = torch.tensor([labels.size(1)])

    combined, prior_loss, posterior_loss = plan_w1_loss_for_gates(
        teacher_plan=teacher_plan,
        effective_log_probs=posterior_log_probs,
        prior_log_probs=prior_log_probs,
        posterior_log_probs=posterior_log_probs,
        labels=labels,
        frame_lens=frame_lens,
        label_lens=label_lens,
        dual_gate=True,
        prior_weight=0.5,
        posterior_weight=0.5,
    )
    assert torch.allclose(combined, 0.5 * (prior_loss + posterior_loss))
    combined.sum().backward()
    assert teacher_plan.grad is None
    assert prior_logits.grad is not None
    assert prior_logits.grad.abs().sum() > 0
    assert posterior_logits.grad is not None
    assert posterior_logits.grad.abs().sum() > 0


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


if __name__ == "__main__":
    test_gated_log_probs_normalized()
    test_cross_attention_gate_masks_padding_frames()
    test_dual_gate_supervision_trains_both_gates()
    test_blank_kl_distills_posterior_to_prior_only()
    test_dual_gate_plan_w1_trains_prior_and_posterior_not_plan()
    test_vi_ot_loss_v2_gradient_is_decoupled_from_gate()
    test_vi_ot_loss_v2_acoustic_column_marginal()
    test_vi_ot_loss_v2_batched_matches_loop_acoustic()
    test_vi_ot_loss_v2_batched_matches_loop_bpe()
    test_vi_ot_loss_v2_batched_matches_loop_uniform()
    test_vi_ot_loss_v2_batched_gradient_is_decoupled_from_gate()
    print("All tests passed.")
