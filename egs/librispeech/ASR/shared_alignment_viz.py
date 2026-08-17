from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


def _make_pos_cost(T: int, U: int, device: torch.device, beta_pos: float) -> Optional[torch.Tensor]:
    if beta_pos <= 0 or T == 0 or U == 0:
        return None
    tpos = torch.linspace(0, 1, T, device=device).unsqueeze(1)
    upos = torch.linspace(0, 1, U, device=device).unsqueeze(0)
    return beta_pos * (tpos - upos).pow(2)


def _sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    eps: float = 0.1,
    iters: int = 50,
) -> torch.Tensor:
    tiny = 1e-8
    eps = max(float(eps), tiny)

    log_a = torch.log(a.clamp_min(tiny))
    log_b = torch.log(b.clamp_min(tiny))
    log_K = -C / eps

    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(
            log_K.transpose(0, 1) + log_u.unsqueeze(0), dim=1
        )

    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    return torch.exp(log_P)


def _compute_ctc_state_occupancy(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    log_probs = log_probs.to(torch.float64)
    T = log_probs.size(0)
    U = labels.numel()
    if T == 0 or U == 0:
        return log_probs.new_zeros((T, U), dtype=torch.float32)

    ext_labels = labels.new_full((2 * U + 1,), blank_id)
    ext_labels[1::2] = labels
    S = ext_labels.numel()
    emit = log_probs.index_select(1, ext_labels)

    alpha = log_probs.new_full((T, S), float("-inf"))
    alpha[0, 0] = emit[0, 0]
    if S > 1:
        alpha[0, 1] = emit[0, 1]

    neg = log_probs.new_full((S,), float("-inf"))
    skip_forward = torch.zeros(S, dtype=torch.bool, device=log_probs.device)
    if S > 2:
        skip_forward[2:] = (
            (ext_labels[2:] != blank_id)
            & (ext_labels[2:] != ext_labels[:-2])
        )

    for t in range(1, T):
        prev = alpha[t - 1]
        stay = prev
        step = torch.cat((neg[:1], prev[:-1]))
        skip = torch.cat((neg[:2], prev[:-2])).masked_fill(
            ~skip_forward, float("-inf")
        )
        alpha[t] = emit[t] + torch.logsumexp(
            torch.stack((stay, step, skip), dim=0), dim=0
        )

    beta = log_probs.new_full((T, S), float("-inf"))
    beta[T - 1, S - 1] = emit[T - 1, S - 1]
    if S > 1:
        beta[T - 1, S - 2] = emit[T - 1, S - 2]

    skip_backward = torch.zeros(S, dtype=torch.bool, device=log_probs.device)
    if S > 2:
        skip_backward[:-2] = (
            (ext_labels[2:] != blank_id)
            & (ext_labels[2:] != ext_labels[:-2])
        )

    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1]
        stay = nxt
        step = torch.cat((nxt[1:], neg[:1]))
        skip = torch.cat((nxt[2:], neg[:2])).masked_fill(
            ~skip_backward, float("-inf")
        )
        beta[t] = emit[t] + torch.logsumexp(
            torch.stack((stay, step, skip), dim=0), dim=0
        )

    log_Z = torch.logsumexp(alpha[T - 1, max(S - 2, 0) : S], dim=0)
    posterior = torch.exp(alpha + beta - emit - log_Z)
    return posterior[:, 1::2].to(torch.float32)


def _barycentric_curve(alignment: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if alignment is None or alignment.numel() == 0:
        return None
    positions = torch.arange(
        alignment.size(1), device=alignment.device, dtype=alignment.dtype
    )
    row_mass = alignment.sum(dim=1).clamp_min(1e-8)
    return (alignment * positions.unsqueeze(0)).sum(dim=1) / row_mass


def _support_width(
    alignment: Optional[torch.Tensor],
    relative_threshold: float,
) -> Optional[torch.Tensor]:
    if alignment is None or alignment.numel() == 0:
        return None
    peaks = alignment.max(dim=0).values
    active = peaks > 0
    threshold = peaks * relative_threshold
    support = (alignment >= threshold.unsqueeze(0)).sum(dim=0)
    return torch.where(active, support, torch.zeros_like(support))


def _argmax_nonblank_run_stats(
    argmax_ids: torch.Tensor,
    blank_id: int,
) -> Dict[str, float]:
    widths: List[int] = []
    current_token: Optional[int] = None
    current_width = 0

    for token in argmax_ids.detach().cpu().tolist():
        token = int(token)
        if token == blank_id:
            if current_width > 0:
                widths.append(current_width)
            current_token = None
            current_width = 0
            continue

        if current_token is None or token != current_token:
            if current_width > 0:
                widths.append(current_width)
            current_token = token
            current_width = 1
        else:
            current_width += 1

    if current_width > 0:
        widths.append(current_width)

    if not widths:
        return {
            "spike_width_mean": 0.0,
            "spike_width_max": 0.0,
            "spike_run_count": 0.0,
        }

    width_tensor = torch.tensor(widths, dtype=torch.float32)
    return {
        "spike_width_mean": float(width_tensor.mean().item()),
        "spike_width_max": float(width_tensor.max().item()),
        "spike_run_count": float(width_tensor.numel()),
    }


def compute_alignment_stats(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    token_pieces: Sequence[str],
    blank_id: int = 0,
    tau: float = 0.1,
    eps: float = 0.5,
    iters: int = 40,
    beta_pos: float = 1.0,
    support_relative_threshold: float = 0.1,
    ot_coupling_override: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    def _to_cpu_or_none(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if value is None else value.cpu()

    log_probs = log_probs.detach().float()
    labels = labels.detach().long()
    T = log_probs.size(0)
    U = labels.numel()
    probs = log_probs.exp()

    p_blank = log_probs[:, blank_id].exp()
    frame_entropy = -(probs * log_probs).sum(dim=-1)
    argmax_prob, argmax_ids = probs.max(dim=-1)
    argmax_nonblank = argmax_ids != blank_id
    spike_stats = _argmax_nonblank_run_stats(argmax_ids, blank_id=blank_id)
    keep = (1.0 - p_blank) > tau
    if keep.sum().item() < 2 and T >= 2:
        topk = torch.topk(1.0 - p_blank, k=2).indices
        keep = torch.zeros_like(keep)
        keep[topk] = True
    elif keep.sum().item() == 0 and T > 0:
        keep = torch.ones_like(keep, dtype=torch.bool)

    ctc_occupancy = _compute_ctc_state_occupancy(log_probs, labels, blank_id)
    ctc_occupancy_kept = ctc_occupancy[keep]

    token_log_probs = log_probs.gather(
        dim=1, index=labels.unsqueeze(0).expand(T, U)
    )
    token_posteriors_kept = token_log_probs[keep].exp()

    log_probs_kept = log_probs[keep]
    T2 = int(log_probs_kept.size(0))

    if ot_coupling_override is not None:
        ot_coupling = ot_coupling_override.detach().float().to(log_probs.device)
        if ot_coupling.shape == (T, U):
            ot_coupling = ot_coupling[keep]
        elif ot_coupling.shape != (T2, U):
            raise ValueError(
                "ot_coupling_override must have shape "
                f"{(T, U)} or {(T2, U)}, got {tuple(ot_coupling.shape)}"
            )
    elif T2 > 0 and U > 0:
        D = -log_probs_kept.gather(dim=1, index=labels.unsqueeze(0).expand(T2, U))
        pos_cost = _make_pos_cost(T2, U, device=log_probs.device, beta_pos=beta_pos)
        if pos_cost is not None:
            D = D + pos_cost

        a = torch.full((T2,), 1.0 / T2, device=log_probs.device)
        b = torch.full((U,), 1.0 / U, device=log_probs.device)
        ot_coupling = _sinkhorn(a, b, D.detach(), eps=eps, iters=iters).to(torch.float32)
    else:
        ot_coupling = log_probs.new_zeros((T2, U), dtype=torch.float32)

    stats = {
        "keep": keep.cpu(),
        "kept_frame_indices": torch.nonzero(keep, as_tuple=False).squeeze(-1).cpu(),
        "blank_prob": p_blank.cpu(),
        "nonblank_prob": (1.0 - p_blank).cpu(),
        "frame_entropy": frame_entropy.cpu(),
        "argmax_ids": argmax_ids.cpu(),
        "argmax_prob": argmax_prob.cpu(),
        "argmax_nonblank": argmax_nonblank.cpu(),
        "mean_blank_prob": float(p_blank.mean().item()) if T > 0 else 0.0,
        "mean_nonblank_prob": float((1.0 - p_blank).mean().item()) if T > 0 else 0.0,
        "mean_frame_entropy": float(frame_entropy.mean().item()) if T > 0 else 0.0,
        "posterior_entropy": float(frame_entropy.mean().item()) if T > 0 else 0.0,
        "posterior_peakiness": float(argmax_prob.mean().item()) if T > 0 else 0.0,
        "argmax_nonblank_ratio": float(argmax_nonblank.float().mean().item())
        if T > 0
        else 0.0,
        **spike_stats,
        "labels": labels.cpu(),
        "token_pieces": list(token_pieces),
        "token_posteriors_kept": token_posteriors_kept.cpu(),
        "ctc_occupancy_kept": ctc_occupancy_kept.cpu(),
        "ctc_occupancy": ctc_occupancy.cpu(),
        "ot_coupling": ot_coupling.cpu(),
        "ctc_barycentric": _to_cpu_or_none(_barycentric_curve(ctc_occupancy_kept)),
        "ot_barycentric": _to_cpu_or_none(_barycentric_curve(ot_coupling)),
        "ctc_support": _to_cpu_or_none(
            _support_width(
                ctc_occupancy_kept, relative_threshold=support_relative_threshold
            )
        ),
        "ot_support": _to_cpu_or_none(
            _support_width(ot_coupling, relative_threshold=support_relative_threshold)
        ),
        "support_relative_threshold": support_relative_threshold,
        "tau": tau,
        "eps": eps,
        "iters": iters,
        "beta_pos": beta_pos,
    }
    return stats


def compute_plan_agreement_metrics(
    ctc_alignment: torch.Tensor,
    ot_alignment: torch.Tensor,
    support_relative_threshold: float = 0.1,
) -> Dict[str, float]:
    """Compare two frame-token alignments after normalizing their total mass."""
    ctc = ctc_alignment.detach().float()
    ot = ot_alignment.detach().float().to(ctc.device)
    if ctc.shape != ot.shape:
        raise ValueError(
            f"Alignment shapes must match, got {tuple(ctc.shape)} and {tuple(ot.shape)}"
        )
    if ctc.numel() == 0:
        return {
            "plan_ctc_barycenter_mad": 0.0,
            "plan_ctc_support_iou": 0.0,
            "plan_ctc_total_variation": 0.0,
        }

    tiny = 1.0e-8
    t_pos = torch.linspace(0, 1, ctc.size(0), device=ctc.device).unsqueeze(1)
    ctc_col_mass = ctc.sum(dim=0)
    ot_col_mass = ot.sum(dim=0)
    valid = (ctc_col_mass > tiny) & (ot_col_mass > tiny)
    if valid.any():
        ctc_center = (ctc * t_pos).sum(dim=0) / ctc_col_mass.clamp_min(tiny)
        ot_center = (ot * t_pos).sum(dim=0) / ot_col_mass.clamp_min(tiny)
        barycenter_mad = (ctc_center[valid] - ot_center[valid]).abs().mean()
    else:
        barycenter_mad = ctc.new_tensor(0.0)

    ctc_peak = ctc.max(dim=0).values
    ot_peak = ot.max(dim=0).values
    ctc_support = (ctc >= ctc_peak.mul(support_relative_threshold).unsqueeze(0)) & (
        ctc_peak > 0
    ).unsqueeze(0)
    ot_support = (ot >= ot_peak.mul(support_relative_threshold).unsqueeze(0)) & (
        ot_peak > 0
    ).unsqueeze(0)
    union = (ctc_support | ot_support).sum().float()
    intersection = (ctc_support & ot_support).sum().float()
    support_iou = intersection / union.clamp_min(1.0)

    ctc_norm = ctc / ctc.sum().clamp_min(tiny)
    ot_norm = ot / ot.sum().clamp_min(tiny)
    total_variation = 0.5 * (ctc_norm - ot_norm).abs().sum()
    return {
        "plan_ctc_barycenter_mad": float(barycenter_mad.item()),
        "plan_ctc_support_iou": float(support_iou.item()),
        "plan_ctc_total_variation": float(total_variation.item()),
    }


def _alignment_diagonal_metrics(
    alignment: Optional[torch.Tensor],
    diagonal_band_width: float,
) -> Dict[str, float]:
    if alignment is None or alignment.numel() == 0:
        return {
            "diag_mean_abs_dev": 0.0,
            "offdiag_mass": 0.0,
        }

    T, U = alignment.shape
    if T == 0 or U == 0:
        return {
            "diag_mean_abs_dev": 0.0,
            "offdiag_mass": 0.0,
        }

    total_mass = alignment.sum().clamp_min(1e-8)
    tpos = torch.linspace(0, 1, T, device=alignment.device, dtype=alignment.dtype)
    upos = torch.linspace(0, 1, U, device=alignment.device, dtype=alignment.dtype)
    dist = (tpos.unsqueeze(1) - upos.unsqueeze(0)).abs()

    diag_mean_abs_dev = float(((alignment * dist).sum() / total_mass).item())
    offdiag_mask = dist > diagonal_band_width
    offdiag_mass = float((alignment[offdiag_mask].sum() / total_mass).item())
    return {
        "diag_mean_abs_dev": diag_mean_abs_dev,
        "offdiag_mass": offdiag_mass,
    }


def _barycentric_curve_metrics(
    curve: Optional[torch.Tensor],
    num_tokens: int,
    backward_tol: float,
) -> Dict[str, float]:
    if curve is None or curve.numel() < 2:
        return {
            "bary_jitter": 0.0,
            "backward_rate": 0.0,
        }

    diffs = curve[1:] - curve[:-1]
    backward_rate = float((diffs < -backward_tol).float().mean().item())
    if diffs.numel() < 2:
        return {
            "bary_jitter": 0.0,
            "backward_rate": backward_rate,
        }

    scale = max(float(num_tokens - 1), 1.0)
    second_diff = diffs[1:] - diffs[:-1]
    bary_jitter = float(second_diff.abs().mean().item() / scale)
    return {
        "bary_jitter": bary_jitter,
        "backward_rate": backward_rate,
    }


def compute_alignment_quality_metrics(
    stats: Dict[str, Any],
    diagonal_band_width: float = 0.12,
    backward_tol: float = 0.05,
) -> Dict[str, float]:
    num_frames = int(stats["blank_prob"].numel())
    num_kept_frames = int(stats["kept_frame_indices"].numel())
    keep_ratio = float(num_kept_frames / max(num_frames, 1))

    ctc_diag_metrics = _alignment_diagonal_metrics(
        stats.get("ctc_occupancy_kept"),
        diagonal_band_width=diagonal_band_width,
    )
    ot_diag_metrics = _alignment_diagonal_metrics(
        stats.get("ot_coupling"),
        diagonal_band_width=diagonal_band_width,
    )
    ctc_curve_metrics = _barycentric_curve_metrics(
        stats.get("ctc_barycentric"),
        num_tokens=len(stats["token_pieces"]),
        backward_tol=backward_tol,
    )
    ot_curve_metrics = _barycentric_curve_metrics(
        stats.get("ot_barycentric"),
        num_tokens=len(stats["token_pieces"]),
        backward_tol=backward_tol,
    )

    ctc_support = stats.get("ctc_support")
    ot_support = stats.get("ot_support")
    num_kept = max(num_kept_frames, 1)

    metrics = {
        "num_frames": float(num_frames),
        "num_kept_frames": float(num_kept_frames),
        "keep_ratio": keep_ratio,
        "mean_blank_prob": float(stats["mean_blank_prob"]),
        "mean_nonblank_prob": float(stats["mean_nonblank_prob"]),
        "mean_frame_entropy": float(stats["mean_frame_entropy"]),
        "posterior_entropy": float(stats["posterior_entropy"]),
        "posterior_peakiness": float(stats["posterior_peakiness"]),
        "argmax_nonblank_ratio": float(stats["argmax_nonblank_ratio"]),
        "spike_width_mean": float(stats["spike_width_mean"]),
        "spike_width_max": float(stats["spike_width_max"]),
        "spike_run_count": float(stats["spike_run_count"]),
        "ctc_diag_mean_abs_dev": ctc_diag_metrics["diag_mean_abs_dev"],
        "ctc_offdiag_mass": ctc_diag_metrics["offdiag_mass"],
        "ot_diag_mean_abs_dev": ot_diag_metrics["diag_mean_abs_dev"],
        "ot_offdiag_mass": ot_diag_metrics["offdiag_mass"],
        "ctc_bary_jitter": ctc_curve_metrics["bary_jitter"],
        "ctc_backward_rate": ctc_curve_metrics["backward_rate"],
        "ot_bary_jitter": ot_curve_metrics["bary_jitter"],
        "ot_backward_rate": ot_curve_metrics["backward_rate"],
        "ctc_support_mean": float(ctc_support.float().mean().item())
        if ctc_support is not None and ctc_support.numel() > 0
        else 0.0,
        "ot_support_mean": float(ot_support.float().mean().item())
        if ot_support is not None and ot_support.numel() > 0
        else 0.0,
        "ctc_support_ratio": float(ctc_support.float().mean().item() / num_kept)
        if ctc_support is not None and ctc_support.numel() > 0
        else 0.0,
        "ot_support_ratio": float(ot_support.float().mean().item() / num_kept)
        if ot_support is not None and ot_support.numel() > 0
        else 0.0,
    }
    return metrics


def save_alignment_comparison_figure(
    reports: Sequence[Dict[str, Any]],
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not reports:
        raise ValueError("reports must not be empty")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ctc_vmax = max(
        float(r["stats"]["ctc_occupancy_kept"].max().item())
        if r["stats"]["ctc_occupancy_kept"].numel() > 0
        else 0.0
        for r in reports
    )
    ot_vmax = max(
        float(r["stats"]["ot_coupling"].max().item())
        if r["stats"]["ot_coupling"].numel() > 0
        else 0.0
        for r in reports
    )
    ctc_vmax = max(ctc_vmax, 1e-6)
    ot_vmax = max(ot_vmax, 1e-6)

    nrows = len(reports)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=6,
        figsize=(29, 4.6 * nrows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.25, 1.25, 1.0, 1.05, 0.95, 0.95]},
    )

    col_titles = [
        "CTC State Occupancy",
        "OT Coupling",
        "Barycentric Alignment",
        "Per-Token Support",
        "Frame-wise Entropy",
        "Blank Probability",
    ]
    for col_idx, col_title in enumerate(col_titles):
        axes[0, col_idx].set_title(col_title)

    for row_idx, report in enumerate(reports):
        stats = report["stats"]
        token_pieces = stats["token_pieces"]
        num_tokens = len(token_pieces)
        ctc = stats["ctc_occupancy_kept"].numpy()
        ot = stats["ot_coupling"].numpy()
        ctc_bar = (
            stats["ctc_barycentric"].numpy()
            if stats["ctc_barycentric"] is not None
            else np.zeros((ctc.shape[0],), dtype=np.float32)
        )
        ot_bar = (
            stats["ot_barycentric"].numpy()
            if stats["ot_barycentric"] is not None
            else np.zeros((ot.shape[0],), dtype=np.float32)
        )
        ctc_support = (
            stats["ctc_support"].numpy()
            if stats["ctc_support"] is not None
            else np.zeros((num_tokens,), dtype=np.float32)
        )
        ot_support = (
            stats["ot_support"].numpy()
            if stats["ot_support"] is not None
            else np.zeros((num_tokens,), dtype=np.float32)
        )
        frame_entropy = stats["frame_entropy"].numpy()
        blank_prob = stats["blank_prob"].numpy()
        nonblank_prob = stats["nonblank_prob"].numpy()
        kept_frame_indices = stats["kept_frame_indices"].numpy()

        ax = axes[row_idx, 0]
        im = ax.imshow(ctc.T, aspect="auto", origin="lower", vmin=0.0, vmax=ctc_vmax)
        ax.set_ylabel(f"{report['name']} Model\nToken")
        ax.set_xlabel("Kept frame index")
        if num_tokens <= 60:
            ax.set_yticks(range(num_tokens))
            ax.set_yticklabels(token_pieces)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row_idx, 1]
        im = ax.imshow(ot.T, aspect="auto", origin="lower", vmin=0.0, vmax=ot_vmax)
        ax.set_xlabel("Kept frame index")
        if num_tokens <= 60:
            ax.set_yticks(range(num_tokens))
            ax.set_yticklabels(token_pieces)
        else:
            ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row_idx, 2]
        x_ctc = np.arange(len(ctc_bar))
        x_ot = np.arange(len(ot_bar))
        ax.plot(x_ctc, ctc_bar, label="CTC", color="#4C78A8", linewidth=2.0)
        ax.plot(x_ot, ot_bar, label="OT", color="#F58518", linewidth=2.0)
        ax.set_xlabel("Kept frame index")
        ax.set_ylabel("Token index")
        ax.set_ylim(-0.5, max(num_tokens - 0.5, 0.5))
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(loc="upper left")

        ax = axes[row_idx, 3]
        x = np.arange(num_tokens)
        width = 0.42
        ax.bar(x - width / 2, ctc_support, width=width, label="CTC", color="#4C78A8")
        ax.bar(x + width / 2, ot_support, width=width, label="OT", color="#F58518")
        ax.set_xlabel("Token")
        ax.set_ylabel("Support width")
        ax.legend(loc="upper right")
        if num_tokens <= 60:
            ax.set_xticks(x)
            ax.set_xticklabels(token_pieces, rotation=90)
        else:
            ax.set_xticks([])

        ax = axes[row_idx, 4]
        x_full = np.arange(len(frame_entropy))
        ax.plot(x_full, frame_entropy, color="#54A24B", linewidth=2.0)
        if kept_frame_indices.size > 0:
            ax.scatter(
                kept_frame_indices,
                frame_entropy[kept_frame_indices],
                s=12,
                color="#2F4B7C",
                alpha=0.8,
                label="Kept",
            )
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Entropy (nats)")
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_title(
            f"mean={stats['mean_frame_entropy']:.2f}",
            fontsize=10,
            pad=6,
        )
        if kept_frame_indices.size > 0:
            ax.legend(loc="upper right")

        ax = axes[row_idx, 5]
        ax.plot(x_full, blank_prob, color="#E45756", linewidth=2.0, label="Blank")
        ax.plot(
            x_full,
            nonblank_prob,
            color="#72B7B2",
            linewidth=1.8,
            alpha=0.9,
            label="Non-blank",
        )
        if kept_frame_indices.size > 0:
            ax.scatter(
                kept_frame_indices,
                blank_prob[kept_frame_indices],
                s=10,
                color="#B279A2",
                alpha=0.7,
            )
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_title(
            f"mean blank={stats['mean_blank_prob']:.2f}",
            fontsize=10,
            pad=6,
        )
        ax.legend(loc="upper right")

    if title is not None:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_alignment_report(
    reports: Sequence[Dict[str, Any]],
    out_prefix: Path,
    title: Optional[str] = None,
) -> Dict[str, Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pt_path = out_prefix.with_suffix(".pt")
    png_path = out_prefix.with_suffix(".png")
    torch.save(list(reports), pt_path)
    save_alignment_comparison_figure(reports=reports, out_path=png_path, title=title)
    return {"pt": pt_path, "png": png_path}
