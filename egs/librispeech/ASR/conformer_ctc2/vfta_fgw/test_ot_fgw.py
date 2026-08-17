#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

_RECIPE_DIR = Path(__file__).resolve().parent.parent
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

from ot_prior_v2 import vi_ot_loss_v2  # noqa: E402
from vfta_fgw.ot_fgw import (  # noqa: E402
    DiagonalResidualPSDFrameMetric,
    LearnablePSDFrameMetric,
    _compute_dx,
    metric_occupancy_geometry_loss,
    vi_fgw_loss_v2,
)


def test_lambda_zero_matches_vfta_ot() -> None:
    torch.manual_seed(0)
    T, U, Vm1 = 12, 5, 9
    logits = torch.randn(T, Vm1, requires_grad=True)
    log_p_nonblank = torch.log_softmax(logits, dim=-1)
    alpha = torch.sigmoid(torch.randn(T))
    labels = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)
    acoustic = torch.randn(T, 80)

    kwargs = dict(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        column_marginal_type="acoustic",
        alpha_smooth_mix=0.1,
        token_prior_sigma=0.15,
        token_prior_score_temp=1.0,
        token_prior_floor=0.05,
        eps=0.3,
        iters=7,
        beta_pos=1.0,
    )
    base = vi_ot_loss_v2(**kwargs)
    fgw_zero = vi_fgw_loss_v2(
        **kwargs,
        acoustic_features=acoustic,
        lambda_gw=0.0,
        n_outer=2,
    )
    assert torch.allclose(base, fgw_zero, atol=1.0e-6), (base, fgw_zero)


def test_fgw_plan_and_backward() -> None:
    torch.manual_seed(1)
    T, U, Vm1 = 14, 6, 10
    logits = torch.randn(T, Vm1, requires_grad=True)
    log_p_nonblank = torch.log_softmax(logits, dim=-1)
    alpha = torch.sigmoid(torch.randn(T))
    labels = torch.tensor([1, 2, 4, 6, 8, 10], dtype=torch.long)
    acoustic = torch.randn(T, 80)

    loss, plan = vi_fgw_loss_v2(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        acoustic_features=acoustic,
        column_marginal_type="acoustic",
        alpha_smooth_mix=0.1,
        token_prior_sigma=0.15,
        token_prior_score_temp=1.0,
        token_prior_floor=0.05,
        eps=0.3,
        iters=7,
        beta_pos=1.0,
        lambda_gw=0.1,
        n_outer=2,
        return_plan=True,
    )

    assert plan is not None
    assert plan.shape == (T, U)
    assert torch.isfinite(loss)
    assert torch.isfinite(plan).all()
    assert torch.allclose(plan.sum(), torch.tensor(1.0), atol=2.0e-4)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_learnable_metric_identity_matches_fixed_cosine() -> None:
    torch.manual_seed(2)
    acoustic = torch.randn(17, 80)
    rng_before = torch.random.get_rng_state().clone()
    metric = LearnablePSDFrameMetric(feature_dim=80)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    learned, diagnostics = metric(acoustic)
    fixed = _compute_dx(acoustic)
    assert torch.allclose(learned, fixed, atol=2.0e-6), (
        (learned - fixed).abs().max()
    )
    assert diagnostics["metric_moment_reg"].abs() < 1.0e-10
    assert diagnostics["metric_identity_reg"].abs() < 1.0e-10


def test_diagonal_residual_metric_identity_and_rho_control() -> None:
    torch.manual_seed(21)
    acoustic = torch.randn(17, 80)
    fixed = _compute_dx(acoustic)
    metric = DiagonalResidualPSDFrameMetric(
        feature_dim=80, max_log_scale=0.5
    )

    initial, diagnostics = metric(acoustic, rho=1.0)
    assert torch.allclose(initial, fixed, atol=2.0e-6)
    assert diagnostics["metric_identity_reg"].abs() < 1.0e-10

    with torch.no_grad():
        metric.projection.weight.copy_(torch.linspace(-4.0, 4.0, 80))
    fixed_endpoint, _ = metric(acoustic, rho=0.0)
    learned_endpoint, _ = metric(acoustic, rho=1.0)
    assert torch.allclose(fixed_endpoint, fixed, atol=2.0e-6)
    assert not torch.allclose(learned_endpoint, fixed, atol=1.0e-4)

    factor = metric.effective_projection_weight(rho=1.0)
    eigenvalues = factor.diagonal().square()
    assert eigenvalues.min() >= torch.exp(torch.tensor(-1.0)) - 1.0e-6
    assert eigenvalues.max() <= torch.exp(torch.tensor(1.0)) + 1.0e-6


def test_selective_metric_gradient_and_plan_equivalence() -> None:
    torch.manual_seed(3)
    T, U, Vm1 = 16, 7, 11
    base_logits = torch.randn(T, Vm1)
    logits_fixed = base_logits.clone().requires_grad_(True)
    logits_learned = base_logits.clone().requires_grad_(True)
    alpha = torch.sigmoid(torch.randn(T))
    labels = torch.tensor([1, 2, 4, 5, 7, 9, 11], dtype=torch.long)
    acoustic_fixed = torch.randn(T, 80)
    acoustic_learned = acoustic_fixed.clone().requires_grad_(True)
    metric = LearnablePSDFrameMetric(feature_dim=80)

    common = dict(
        alpha=alpha,
        labels=labels,
        bpe_lengths=None,
        column_marginal_type="acoustic",
        alpha_smooth_mix=0.1,
        token_prior_sigma=0.15,
        token_prior_score_temp=1.0,
        token_prior_floor=0.05,
        eps=0.3,
        iters=9,
        beta_pos=1.0,
        lambda_gw=0.1,
        n_outer=3,
        return_plan=True,
    )
    fixed_loss, fixed_plan = vi_fgw_loss_v2(
        log_p_nonblank=torch.log_softmax(logits_fixed, dim=-1),
        acoustic_features=acoustic_fixed,
        **common,
    )
    learned_loss, learned_plan = vi_fgw_loss_v2(
        log_p_nonblank=torch.log_softmax(logits_learned, dim=-1),
        acoustic_features=acoustic_learned,
        frame_metric=metric,
        metric_grad_scale=1.0,
        metric_moment_reg_weight=0.0,
        metric_identity_reg_weight=0.0,
        **common,
    )

    assert fixed_plan is not None and learned_plan is not None
    assert torch.allclose(fixed_plan, learned_plan, atol=3.0e-6), (
        (fixed_plan - learned_plan).abs().max()
    )
    assert torch.allclose(fixed_loss, learned_loss.detach(), atol=2.0e-6)

    fixed_loss.backward()
    learned_loss.backward()
    assert torch.allclose(logits_fixed.grad, logits_learned.grad, atol=3.0e-6)
    metric_grad = metric.projection.weight.grad
    assert metric_grad is not None
    assert torch.isfinite(metric_grad).all()
    assert metric_grad.abs().sum() > 0
    # Structural input is deliberately detached; only L receives this path.
    assert acoustic_learned.grad is None


def test_metric_regularizer_detects_collapse() -> None:
    torch.manual_seed(4)
    metric = LearnablePSDFrameMetric(feature_dim=80)
    with torch.no_grad():
        metric.projection.weight.zero_()
    _, diagnostics = metric(torch.randn(12, 80))
    assert diagnostics["metric_moment_reg"] > 0
    assert diagnostics["metric_identity_reg"] > 0


def test_metric_warmup_dependency_has_exact_zero_gradient() -> None:
    torch.manual_seed(5)
    logits = torch.randn(13, 9, requires_grad=True)
    metric = LearnablePSDFrameMetric(feature_dim=80)
    loss = vi_fgw_loss_v2(
        log_p_nonblank=torch.log_softmax(logits, dim=-1),
        alpha=torch.sigmoid(torch.randn(13)),
        labels=torch.tensor([1, 2, 4, 6, 9]),
        acoustic_features=torch.randn(13, 80),
        lambda_gw=0.1,
        n_outer=3,
        iters=7,
        frame_metric=metric,
        metric_grad_scale=0.0,
        metric_moment_reg_weight=0.0,
        metric_identity_reg_weight=0.0,
    )
    loss.backward()
    gradient = metric.projection.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) == 0


def test_corrected_metric_occupancy_gradient_is_selective() -> None:
    torch.manual_seed(6)
    T, U, Vm1 = 15, 6, 10
    base_logits = torch.randn(T, Vm1)
    logits_zero = base_logits.clone().requires_grad_(True)
    logits_corrected = base_logits.clone().requires_grad_(True)
    alpha = torch.sigmoid(torch.randn(T))
    labels = torch.tensor([1, 2, 4, 6, 8, 10])
    acoustic = torch.randn(T, 80, requires_grad=True)
    target = torch.softmax(torch.randn(T, U), dim=0)
    metric_zero = DiagonalResidualPSDFrameMetric(feature_dim=80)
    metric_corrected = DiagonalResidualPSDFrameMetric(feature_dim=80)

    common = dict(
        alpha=alpha,
        labels=labels,
        acoustic_features=acoustic,
        lambda_gw=0.1,
        n_outer=3,
        iters=9,
        frame_metric=None,
        metric_surrogate="ctc-occupancy",
        metric_target_occupancy=target,
        metric_moment_reg_weight=0.0,
        metric_identity_reg_weight=0.0,
    )
    zero_loss = vi_fgw_loss_v2(
        log_p_nonblank=torch.log_softmax(logits_zero, dim=-1),
        **{**common, "frame_metric": metric_zero},
        metric_grad_scale=0.0,
    )
    corrected_loss = vi_fgw_loss_v2(
        log_p_nonblank=torch.log_softmax(logits_corrected, dim=-1),
        **{**common, "frame_metric": metric_corrected},
        metric_grad_scale=1.0,
    )
    assert torch.allclose(zero_loss, corrected_loss.detach(), atol=2.0e-6)

    zero_loss.backward()
    corrected_loss.backward()
    assert torch.allclose(logits_zero.grad, logits_corrected.grad, atol=3.0e-6)
    metric_grad = metric_corrected.projection.weight.grad
    assert metric_grad is not None
    assert torch.isfinite(metric_grad).all()
    assert metric_grad.abs().sum() > 0.0
    # Both the acoustic input and the CTC target are fixed on this branch.
    assert acoustic.grad is None
    assert target.grad is None


def test_metric_occupancy_components_and_spectrum_barrier() -> None:
    plan_logits = torch.full((9, 3), -4.0, requires_grad=True)
    target = torch.zeros(9, 3)
    with torch.no_grad():
        target[1, 0] = 1.0
        target[4, 1] = 1.0
        target[7, 2] = 1.0
    plan = torch.softmax(plan_logits, dim=0)
    loss, diagnostics = metric_occupancy_geometry_loss(plan, target)
    assert torch.isfinite(loss)
    assert diagnostics["metric_occ_w1"] > 0.0
    assert diagnostics["metric_occ_barycenter"] > 0.0
    loss.backward()
    assert plan_logits.grad is not None
    assert torch.isfinite(plan_logits.grad).all()
    assert plan_logits.grad.abs().sum() > 0.0

    metric = LearnablePSDFrameMetric(feature_dim=8)
    identity_reg, identity_diag = metric.spectrum_regularizer()
    assert identity_reg.abs() < 1.0e-6
    assert torch.allclose(identity_diag["metric_eigen_min"], torch.tensor(1.0))
    with torch.no_grad():
        metric.projection.weight[0].mul_(1.0e-3)
        metric.projection.weight[1].mul_(3.0)
    bad_reg, bad_diag = metric.spectrum_regularizer(max_eigenvalue=4.0)
    assert bad_reg > 0.0
    assert bad_diag["metric_eigen_min"] < 1.0e-4
    assert bad_diag["metric_eigen_max"] > 4.0


if __name__ == "__main__":
    test_lambda_zero_matches_vfta_ot()
    test_fgw_plan_and_backward()
    test_learnable_metric_identity_matches_fixed_cosine()
    test_diagonal_residual_metric_identity_and_rho_control()
    test_selective_metric_gradient_and_plan_equivalence()
    test_metric_regularizer_detects_collapse()
    test_metric_warmup_dependency_has_exact_zero_gradient()
    test_corrected_metric_occupancy_gradient_is_selective()
    test_metric_occupancy_components_and_spectrum_barrier()
    print("vfta_fgw ot_fgw smoke OK")
