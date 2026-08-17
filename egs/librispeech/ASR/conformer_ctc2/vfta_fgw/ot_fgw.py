from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ot_prior_v2 import (
    _make_alpha_row_marginal,
    _make_bpe_length_column_marginal,
    _make_soft_column_marginal,
    _make_uniform_column_marginal,
    _sinkhorn,
    vi_ot_loss_v2,
)


@torch.no_grad()
def _make_pos_cost(T: int, U: int, device: torch.device, beta_pos: float) -> Optional[Tensor]:
    if beta_pos <= 0:
        return None
    t_pos = torch.linspace(0, 1, T, device=device).unsqueeze(1)
    u_pos = torch.linspace(0, 1, U, device=device).unsqueeze(0)
    return beta_pos * (t_pos - u_pos).pow(2)


@torch.no_grad()
def _compute_dx(acoustic_features: Tensor) -> Tensor:
    """Pairwise cosine distance over encoder-aligned acoustic features."""
    x = F.normalize(acoustic_features.float(), dim=-1)
    return (1.0 - x @ x.transpose(0, 1)).clamp_min(0.0)


class _IdentityProjection(nn.Module):
    """Square linear map initialized without consuming the global RNG."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(feature_dim))

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, self.weight)


class LearnablePSDFrameMetric(nn.Module):
    """Identity-initialized Mahalanobis cost over detached log-Mel frames.

    For L2-normalized frames and ``projection.weight == I``, the squared
    Mahalanobis cost below is exactly cosine distance:

        0.5 * ||L (z_t - z_t')||^2 == 1 - <z_t, z_t'>.

    The induced matrix ``L.T @ L`` is positive semidefinite for every value of
    the unconstrained projection.  Input features are detached deliberately so
    the structural objective cannot reshape the acoustic encoder or the input
    representation through a shortcut gradient.
    """

    def __init__(self, feature_dim: int = 80) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.projection = _IdentityProjection(self.feature_dim)

    def effective_projection_weight(self, rho: float = 1.0) -> Tensor:
        """Return the square factor that induces the reported PSD metric."""
        del rho
        return self.projection.weight.float()

    def forward(
        self,
        acoustic_features: Tensor,
        rho: float = 1.0,
        normalization: str = "none",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if acoustic_features.ndim != 2:
            raise ValueError(
                "acoustic_features must have shape [T, D], got "
                f"{tuple(acoustic_features.shape)}"
            )
        if acoustic_features.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got "
                f"{acoustic_features.size(-1)}"
            )
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError("rho must be between 0 and 1")
        if normalization not in ("none", "offdiag-rms"):
            raise ValueError(
                "normalization must be 'none' or 'offdiag-rms', got "
                f"{normalization!r}"
            )

        z = F.normalize(acoustic_features.detach().float(), dim=-1)
        fixed = (1.0 - z @ z.transpose(0, 1)).clamp_min(0.0)
        projected = self.projection(z)
        squared_norm = projected.square().sum(dim=-1)
        learned = 0.5 * (
            squared_norm.unsqueeze(1)
            + squared_norm.unsqueeze(0)
            - 2.0 * (projected @ projected.transpose(0, 1))
        ).clamp_min(0.0)

        rho_value = float(rho)
        distance = (1.0 - rho_value) * fixed + rho_value * learned
        if normalization == "offdiag-rms" and distance.size(0) > 1:
            offdiag_mask = ~torch.eye(
                distance.size(0), dtype=torch.bool, device=distance.device
            )
            rms = distance[offdiag_mask].square().mean().sqrt().clamp_min(1.0e-6)
            distance = distance / rms.detach()

        if distance.size(0) > 1:
            mask = ~torch.eye(
                distance.size(0), dtype=torch.bool, device=distance.device
            )
            learned_values = learned[mask]
            fixed_values = fixed[mask]
        else:
            learned_values = learned.reshape(-1)
            fixed_values = fixed.reshape(-1)

        moment_reg = (
            learned_values.mean() - fixed_values.detach().mean()
        ).square() + (
            learned_values.std(unbiased=False)
            - fixed_values.detach().std(unbiased=False)
        ).square()
        # Keep the PSD anchor in FP32 under CUDA autocast.
        with torch.autocast(
            device_type=self.projection.weight.device.type, enabled=False
        ):
            weight = self.projection.weight.float()
            gram = weight.transpose(0, 1) @ weight
        identity = torch.eye(
            self.feature_dim, dtype=gram.dtype, device=gram.device
        )
        identity_reg = (gram - identity).square().mean()
        diagnostics = {
            "metric_moment_reg": moment_reg,
            "metric_identity_reg": identity_reg,
            "metric_delta_fro": (weight - identity).square().mean().sqrt(),
        }
        return distance, diagnostics

    def spectrum_regularizer(
        self,
        max_eigenvalue: float = 4.0,
        eps: float = 1.0e-4,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Keep ``L.T @ L`` away from null and exploding directions.

        The log-determinant barrier is zero at the identity (up to floating
        point roundoff), grows sharply near a singular Gram matrix, and is
        complemented by a soft upper-eigenvalue barrier.  It is evaluated once
        per minibatch by the trainer rather than once per utterance.
        """
        if max_eigenvalue <= 1.0:
            raise ValueError("max_eigenvalue must be greater than 1")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        # This small decomposition must remain in FP32 even when the caller is
        # inside CUDA autocast; slogdet/eigvalsh do not support FP16 and the
        # near-null directions are precisely where half precision is unsafe.
        with torch.autocast(
            device_type=self.projection.weight.device.type, enabled=False
        ):
            weight = self.projection.weight.float()
            gram = weight.transpose(0, 1) @ weight
        identity = torch.eye(
            self.feature_dim, dtype=gram.dtype, device=gram.device
        )
        regularized = gram + float(eps) * identity
        _, logdet = torch.linalg.slogdet(regularized)
        identity_value = 1.0 + float(eps)
        identity_barrier = identity_value - torch.log(
            gram.new_tensor(identity_value)
        ) - 1.0
        logdet_barrier = (
            torch.trace(regularized) - logdet - float(self.feature_dim)
        ) / float(self.feature_dim) - identity_barrier

        eigenvalues = torch.linalg.eigvalsh(gram)
        upper_barrier = torch.relu(
            eigenvalues - float(max_eigenvalue)
        ).square().mean()
        total = logdet_barrier + upper_barrier

        eigen_mass = eigenvalues.clamp_min(0.0)
        eigen_prob = eigen_mass / eigen_mass.sum().clamp_min(1.0e-12)
        effective_rank = torch.exp(
            -(eigen_prob * eigen_prob.clamp_min(1.0e-12).log()).sum()
        )
        diagnostics = {
            "metric_spectrum_reg": total,
            "metric_logdet_barrier": logdet_barrier,
            "metric_upper_barrier": upper_barrier,
            "metric_eigen_min": eigenvalues.min(),
            "metric_eigen_max": eigenvalues.max(),
            "metric_effective_rank": effective_rank,
        }
        return total, diagnostics


class _DiagonalResidualProjection(nn.Module):
    """Bounded log-diagonal projection initialized exactly at identity."""

    def __init__(self, feature_dim: int, max_log_scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(feature_dim))
        self.max_log_scale = float(max_log_scale)

    def scales(self, rho: float = 1.0) -> Tensor:
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError("rho must be between 0 and 1")
        return torch.exp(
            float(rho) * self.max_log_scale * torch.tanh(self.weight.float())
        )

    def forward(self, value: Tensor, rho: float = 1.0) -> Tensor:
        return value * self.scales(rho=rho).to(value.dtype)


class DiagonalResidualPSDFrameMetric(nn.Module):
    """Interpretable bounded diagonal Mahalanobis metric.

    The learned factor is L_rho = diag(exp(rho * s * tanh(r))). Thus rho=0
    exactly recovers cosine distance, L_rho.T @ L_rho is always PSD, and each
    log-Mel dimension has a directly readable positive weight.
    """

    def __init__(
        self, feature_dim: int = 80, max_log_scale: float = 0.5
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if max_log_scale <= 0.0:
            raise ValueError("max_log_scale must be positive")
        self.feature_dim = int(feature_dim)
        self.max_log_scale = float(max_log_scale)
        self.projection = _DiagonalResidualProjection(
            feature_dim=self.feature_dim,
            max_log_scale=self.max_log_scale,
        )

    def effective_projection_weight(self, rho: float = 1.0) -> Tensor:
        return torch.diag(self.projection.scales(rho=rho))

    def forward(
        self,
        acoustic_features: Tensor,
        rho: float = 1.0,
        normalization: str = "none",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if acoustic_features.ndim != 2:
            raise ValueError(
                "acoustic_features must have shape [T, D], got "
                f"{tuple(acoustic_features.shape)}"
            )
        if acoustic_features.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got "
                f"{acoustic_features.size(-1)}"
            )
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError("rho must be between 0 and 1")
        if normalization not in ("none", "offdiag-rms"):
            raise ValueError(
                "normalization must be 'none' or 'offdiag-rms', got "
                f"{normalization!r}"
            )

        z = F.normalize(acoustic_features.detach().float(), dim=-1)
        fixed = (1.0 - z @ z.transpose(0, 1)).clamp_min(0.0)
        projected = self.projection(z, rho=rho)
        squared_norm = projected.square().sum(dim=-1)
        distance = 0.5 * (
            squared_norm.unsqueeze(1)
            + squared_norm.unsqueeze(0)
            - 2.0 * (projected @ projected.transpose(0, 1))
        ).clamp_min(0.0)

        if normalization == "offdiag-rms" and distance.size(0) > 1:
            offdiag_mask = ~torch.eye(
                distance.size(0), dtype=torch.bool, device=distance.device
            )
            rms = distance[offdiag_mask].square().mean().sqrt().clamp_min(1.0e-6)
            fixed_rms = fixed[offdiag_mask].square().mean().sqrt().clamp_min(1.0e-6)
            distance = distance * (fixed_rms.detach() / rms.detach())

        if distance.size(0) > 1:
            mask = ~torch.eye(
                distance.size(0), dtype=torch.bool, device=distance.device
            )
            learned_values = distance[mask]
            fixed_values = fixed[mask]
        else:
            learned_values = distance.reshape(-1)
            fixed_values = fixed.reshape(-1)

        moment_reg = (
            learned_values.mean() - fixed_values.detach().mean()
        ).square() + (
            learned_values.std(unbiased=False)
            - fixed_values.detach().std(unbiased=False)
        ).square()
        factor = self.effective_projection_weight(rho=rho)
        gram = factor.transpose(0, 1) @ factor
        identity = torch.eye(
            self.feature_dim, dtype=gram.dtype, device=gram.device
        )
        identity_reg = (gram - identity).square().mean()
        diagnostics = {
            "metric_moment_reg": moment_reg,
            "metric_identity_reg": identity_reg,
            "metric_delta_fro": (factor - identity).square().mean().sqrt(),
        }
        return distance, diagnostics

    def spectrum_regularizer(
        self,
        max_eigenvalue: float = 4.0,
        eps: float = 1.0e-4,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if max_eigenvalue <= 1.0:
            raise ValueError("max_eigenvalue must be greater than 1")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        scales = self.projection.scales(rho=1.0)
        eigenvalues = scales.square()
        regularized = eigenvalues + float(eps)
        identity_value = 1.0 + float(eps)
        identity_barrier = identity_value - torch.log(
            eigenvalues.new_tensor(identity_value)
        ) - 1.0
        logdet_barrier = (
            regularized - regularized.log() - 1.0 - identity_barrier
        ).mean()
        upper_barrier = torch.relu(
            eigenvalues - float(max_eigenvalue)
        ).square().mean()
        total = logdet_barrier + upper_barrier

        eigen_mass = eigenvalues.clamp_min(0.0)
        eigen_prob = eigen_mass / eigen_mass.sum().clamp_min(1.0e-12)
        effective_rank = torch.exp(
            -(eigen_prob * eigen_prob.clamp_min(1.0e-12).log()).sum()
        )
        diagnostics = {
            "metric_spectrum_reg": total,
            "metric_logdet_barrier": logdet_barrier,
            "metric_upper_barrier": upper_barrier,
            "metric_eigen_min": eigenvalues.min(),
            "metric_eigen_max": eigenvalues.max(),
            "metric_effective_rank": effective_rank,
        }
        return total, diagnostics


def metric_occupancy_geometry_loss(
    plan: Tensor,
    target_occupancy: Tensor,
    w1_weight: float = 1.0,
    barycenter_weight: float = 0.5,
    log_std_weight: float = 0.1,
    eps: float = 1.0e-6,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Match a differentiable FGW plan to fixed blank-aware CTC geometry.

    Both inputs have shape ``[T, U]``.  The CTC target is detached defensively;
    gradients can therefore flow only through ``plan``.  W1 and barycenters
    use a [0, 1] time axis so their scale is stable across utterance lengths.
    The log-standard-deviation term explicitly transfers occupancy width.
    """
    if plan.ndim != 2 or target_occupancy.ndim != 2:
        raise ValueError("plan and target_occupancy must both have shape [T, U]")
    if plan.shape != target_occupancy.shape:
        raise ValueError(
            "plan and target_occupancy must have identical shapes: "
            f"{tuple(plan.shape)} != {tuple(target_occupancy.shape)}"
        )
    if min(w1_weight, barycenter_weight, log_std_weight) < 0.0:
        raise ValueError("metric occupancy loss weights must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    num_frames, num_tokens = plan.shape
    if num_frames == 0 or num_tokens == 0:
        zero = plan.new_tensor(0.0)
        return zero, {
            "metric_occ_w1": zero,
            "metric_occ_barycenter": zero,
            "metric_occ_log_std": zero,
            "metric_occ_loss": zero,
        }

    q = plan.float().clamp_min(0.0)
    p = target_occupancy.detach().float().clamp_min(0.0)
    q = q / q.sum(dim=0, keepdim=True).clamp_min(float(eps))
    p = p / p.sum(dim=0, keepdim=True).clamp_min(float(eps))

    time = torch.linspace(0.0, 1.0, num_frames, device=plan.device)[:, None]
    w1 = (q.cumsum(dim=0) - p.cumsum(dim=0)).abs().sum(dim=0)
    w1 = w1 / max(float(num_frames - 1), 1.0)

    q_mean = (q * time).sum(dim=0)
    p_mean = (p * time).sum(dim=0)
    barycenter = (q_mean - p_mean).abs()
    q_std = ((q * (time - q_mean).square()).sum(dim=0) + eps).sqrt()
    p_std = ((p * (time - p_mean).square()).sum(dim=0) + eps).sqrt()
    log_std = (q_std.log() - p_std.log()).abs()

    diagnostics = {
        "metric_occ_w1": w1.mean(),
        "metric_occ_barycenter": barycenter.mean(),
        "metric_occ_log_std": log_std.mean(),
    }
    total = (
        float(w1_weight) * diagnostics["metric_occ_w1"]
        + float(barycenter_weight) * diagnostics["metric_occ_barycenter"]
        + float(log_std_weight) * diagnostics["metric_occ_log_std"]
    )
    diagnostics["metric_occ_loss"] = total
    return total, diagnostics


@torch.no_grad()
def _compute_dy(U: int, device: torch.device) -> Tensor:
    """Pairwise distance over token positions, matching the old FGW recipe."""
    idx = torch.arange(U, device=device, dtype=torch.float32)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() / max(float(U), 1.0)


def _gw_gradient(D_X: Tensor, D_Y: Tensor, A: Tensor) -> Tensor:
    """Gradient of the squared-loss GW energy with respect to A."""
    r = A.sum(dim=1)
    c = A.sum(dim=0)
    term1 = (D_X.pow(2) @ r).unsqueeze(1).expand_as(A)
    term2 = 2.0 * (D_X @ A @ D_Y)
    term3 = (D_Y.pow(2) @ c).unsqueeze(0).expand_as(A)
    return 2.0 * (term1 - term2 + term3)


def _make_column_marginal(
    C_detached: Tensor,
    bpe_lengths: Optional[Tensor],
    column_marginal_type: str,
    bpe_col_floor: float,
    token_prior_sigma: float,
    token_prior_score_temp: float,
    token_prior_floor: float,
) -> Tensor:
    T, U = C_detached.shape
    device = C_detached.device
    if column_marginal_type == "bpe":
        if bpe_lengths is None:
            raise ValueError("bpe column marginal requires bpe_lengths")
        return _make_bpe_length_column_marginal(
            bpe_lengths.to(device),
            floor=bpe_col_floor,
        )
    if column_marginal_type == "acoustic":
        return _make_soft_column_marginal(
            C_detached,
            sigma=token_prior_sigma,
            score_temp=token_prior_score_temp,
            floor=token_prior_floor,
        )
    if column_marginal_type == "uniform":
        return _make_uniform_column_marginal(U, device)
    raise ValueError(f"Unsupported column_marginal_type: {column_marginal_type}")


def vi_fgw_loss_v2(
    log_p_nonblank: Tensor,
    alpha: Tensor,
    labels: Tensor,
    acoustic_features: Tensor,
    bpe_lengths: Optional[Tensor] = None,
    column_marginal_type: str = "acoustic",
    alpha_smooth_mix: float = 0.1,
    bpe_col_floor: float = 0.05,
    token_prior_sigma: float = 0.15,
    token_prior_score_temp: float = 1.0,
    token_prior_floor: float = 0.05,
    eps: float = 0.3,
    iters: int = 30,
    beta_pos: float = 1.0,
    lambda_gw: float = 0.1,
    n_outer: int = 3,
    frame_metric: Optional[nn.Module] = None,
    metric_rho: float = 1.0,
    metric_normalization: str = "none",
    metric_grad_scale: float = 1.0,
    metric_surrogate: str = "token-cost",
    metric_target_occupancy: Optional[Tensor] = None,
    metric_occ_w1_weight: float = 1.0,
    metric_occ_barycenter_weight: float = 0.5,
    metric_occ_log_std_weight: float = 0.1,
    metric_moment_reg_weight: float = 0.1,
    metric_identity_reg_weight: float = 0.01,
    return_plan: bool = False,
) -> Tensor | Tuple[Tensor, Optional[Tensor]]:
    """VFTA OT loss with an optional FGW structural term.

    This is the per-utterance counterpart of vi_ot_loss_v2.  It keeps the VFTA
    cost and marginals unchanged, and only changes the detached transport plan
    when lambda_gw is positive.  Gradients flow through the final <sg[P], C>
    term to the nonblank classifier, exactly as in vi_ot_loss_v2.
    """
    if lambda_gw <= 0.0:
        return vi_ot_loss_v2(
            log_p_nonblank=log_p_nonblank,
            alpha=alpha,
            labels=labels,
            bpe_lengths=bpe_lengths,
            column_marginal_type=column_marginal_type,
            alpha_smooth_mix=alpha_smooth_mix,
            bpe_col_floor=bpe_col_floor,
            token_prior_sigma=token_prior_sigma,
            token_prior_score_temp=token_prior_score_temp,
            token_prior_floor=token_prior_floor,
            eps=eps,
            iters=iters,
            beta_pos=beta_pos,
            return_plan=return_plan,
        )

    lp = log_p_nonblank.float()
    T, vocab_minus_blank = lp.shape
    U = labels.numel()
    device = lp.device

    if U == 0 or T == 0:
        zero = lp.new_tensor(0.0)
        return (zero, None) if return_plan else zero

    if acoustic_features.size(0) != T:
        raise ValueError(
            "acoustic_features must be aligned to log_p_nonblank time axis: "
            f"got {acoustic_features.size(0)} vs {T}"
        )
    if frame_metric is not None and int(n_outer) < 1 and metric_grad_scale > 0:
        raise ValueError(
            "A learnable frame metric requires at least one FGW outer iteration"
        )
    if metric_surrogate not in ("token-cost", "ctc-occupancy"):
        raise ValueError(
            "metric_surrogate must be 'token-cost' or 'ctc-occupancy', got "
            f"{metric_surrogate!r}"
        )

    label_idx = (labels.long() - 1).clamp(min=0, max=vocab_minus_blank - 1)
    C = -lp[:, label_idx]
    C_detached = C.detach()

    pos_cost = _make_pos_cost(T, U, device=device, beta_pos=beta_pos)
    if pos_cost is not None:
        C_detached = C_detached + pos_cost

    a = _make_alpha_row_marginal(
        alpha.detach().float(),
        smooth_mix=alpha_smooth_mix,
    )
    b = _make_column_marginal(
        C_detached=C_detached,
        bpe_lengths=bpe_lengths,
        column_marginal_type=column_marginal_type,
        bpe_col_floor=bpe_col_floor,
        token_prior_sigma=token_prior_sigma,
        token_prior_score_temp=token_prior_score_temp,
        token_prior_floor=token_prior_floor,
    )

    metric_diagnostics: Dict[str, Tensor] = {}
    if frame_metric is None:
        D_X = _compute_dx(acoustic_features.detach())
    else:
        D_X, metric_diagnostics = frame_metric(
            acoustic_features=acoustic_features,
            rho=metric_rho,
            normalization=metric_normalization,
        )
    D_Y = _compute_dy(U, device=device)
    c_scale = C_detached.std().clamp_min(1.0e-6)
    gw_weight = min(max(float(lambda_gw), 0.0), 1.0)

    with torch.no_grad():
        A = _sinkhorn(a, b, C_detached, eps=eps, iters=iters)
        detached_outer = int(n_outer) - (1 if frame_metric is not None else 0)
        for _ in range(max(detached_outer, 0)):
            G = _gw_gradient(D_X, D_Y, A)
            G = G * (c_scale / G.std().clamp_min(1.0e-6))
            C_eff = (1.0 - gw_weight) * C_detached + gw_weight * G
            A = _sinkhorn(a, b, C_eff, eps=eps, iters=iters)

    if frame_metric is not None and int(n_outer) > 0:
        # Unroll only the final conditional-gradient/Sinkhorn update.  The
        # classifier cost, marginals, and previous plan are fixed targets for
        # this path; gradients can therefore reach only the metric parameters.
        G = _gw_gradient(D_X, D_Y, A.detach())
        G_scale = c_scale / G.detach().std().clamp_min(1.0e-6)
        C_eff = (1.0 - gw_weight) * C_detached + gw_weight * G * G_scale
        A = _sinkhorn(a.detach(), b.detach(), C_eff, eps=eps, iters=iters)

    classifier_loss = (A.detach() * C).sum()
    loss = classifier_loss
    if frame_metric is not None:
        metric_loss_active = False
        if float(metric_grad_scale) != 0.0:
            if metric_surrogate == "ctc-occupancy":
                if metric_target_occupancy is None:
                    raise ValueError(
                        "metric_target_occupancy is required for the "
                        "ctc-occupancy metric surrogate"
                    )
                metric_alignment, _ = metric_occupancy_geometry_loss(
                    plan=A,
                    target_occupancy=metric_target_occupancy,
                    w1_weight=metric_occ_w1_weight,
                    barycenter_weight=metric_occ_barycenter_weight,
                    log_std_weight=metric_occ_log_std_weight,
                )
            else:
                metric_alignment = (A * C.detach()).sum()
            # This is exactly zero in the forward pass.  It only supplies the
            # selective plan-to-metric gradient described in the paper design.
            loss = loss + float(metric_grad_scale) * (
                metric_alignment - metric_alignment.detach()
            )
            metric_loss_active = True
        if float(metric_moment_reg_weight) != 0.0:
            loss = loss + float(metric_moment_reg_weight) * metric_diagnostics[
                "metric_moment_reg"
            ]
            metric_loss_active = True
        if float(metric_identity_reg_weight) != 0.0:
            loss = loss + float(metric_identity_reg_weight) * metric_diagnostics[
                "metric_identity_reg"
            ]
            metric_loss_active = True
        if not metric_loss_active:
            # Keep DDP's reducer aware of the parameter during warmup without
            # allowing numerical residue to update the identity metric.
            loss = loss + 0.0 * frame_metric.projection.weight.sum()
    if return_plan:
        return loss, A.detach()
    return loss
