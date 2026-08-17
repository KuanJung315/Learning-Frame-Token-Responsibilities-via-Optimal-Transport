#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import kaldialign
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent
LIBRISPEECH_ASR_DIR = ASR_DIR.parent.parent / "librispeech" / "ASR"
if str(LIBRISPEECH_ASR_DIR) not in sys.path:
    sys.path.append(str(LIBRISPEECH_ASR_DIR))

from ot_fgw import vi_fgw_loss_v2
from ot_prior_v2 import vi_ot_loss_v2
from ctc_plan_consistency import ctc_token_occupancy_batched
from shared_alignment_viz import (
    compute_plan_agreement_metrics,
    compute_alignment_quality_metrics,
    compute_alignment_stats,
)
from timit_eval_common import (
    add_data_arguments,
    baseline_batch_outputs,
    build_baseline_model,
    build_phone_graph,
    build_vi_model,
    get_split_dataloader,
    greedy_runs,
    resolve_device,
    token_ids_to_symbols,
    vi_batch_outputs,
)

from icefall.utils import setup_logger


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("conformer_ctc2/exp_baseline_phone_small/epoch-50.pt"),
    )
    parser.add_argument(
        "--vi-checkpoint",
        type=Path,
        default=Path("conformer_ctc2/exp_vi_ot_phone_small/epoch-50.pt"),
    )
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_phone"))
    parser.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    parser.add_argument("--max-cuts", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--vi-plan-gate",
        choices=["posterior", "prior", "train"],
        default="train",
        help=(
            "Gate used as the plan row marginal. 'train' reconstructs the "
            "final training mixture while keeping CTC emissions on the prior gate."
        ),
    )
    parser.add_argument("--vi-prior-logit-bias", type=float, default=0.0)
    parser.add_argument("--boundary-tolerances-ms", nargs="+", type=float, default=[10, 20, 30, 50])
    parser.add_argument("--shared-ot-tau", type=float, default=0.1)
    parser.add_argument("--shared-ot-eps", type=float, default=None)
    # Override the Sinkhorn eps used to recompute the VI OT plan for boundary
    # extraction (inference-time solver hyperparameter). Lower = sharper plan,
    # less smeared barycenters. None -> use the model's trained ot_eps.
    parser.add_argument("--plan-ot-eps", type=float, default=None)
    parser.add_argument("--shared-ot-iters", type=int, default=None)
    parser.add_argument("--shared-ot-beta-pos", type=float, default=None)
    parser.add_argument("--support-relative-threshold", type=float, default=0.1)
    parser.add_argument("--diagonal-band-width", type=float, default=0.12)
    parser.add_argument("--backward-tol", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/timit_alignment_eval"),
    )
    add_data_arguments(parser)
    return parser


def gold_phone_alignment(cut) -> Dict[str, Any]:
    if len(cut.supervisions) != 1:
        raise ValueError(f"{cut.id}: expected one supervision, got {len(cut.supervisions)}")
    supervision = cut.supervisions[0]
    if not supervision.alignment or "phone" not in supervision.alignment:
        raise ValueError(f"{cut.id}: missing alignment['phone']")

    # Empty symbols are removed by make_phone_cuts.py when it writes text.
    items = [item for item in supervision.alignment["phone"] if item.symbol.strip()]
    symbols = [item.symbol for item in items]
    text_symbols = supervision.text.split()
    if symbols != text_symbols:
        raise ValueError(
            f"{cut.id}: phone alignment/text mismatch: {symbols[:8]} vs {text_symbols[:8]}"
        )

    offset = float(supervision.start)
    centers = [
        offset + float(item.start) + 0.5 * float(item.duration) for item in items
    ]
    starts = [offset + float(item.start) for item in items]
    ends = [
        offset + float(item.start) + float(item.duration) for item in items
    ]
    boundaries = []
    for left, right in zip(items[:-1], items[1:]):
        left_end = float(left.start) + float(left.duration)
        right_start = float(right.start)
        # Removed phones (e.g. TIMIT q -> empty) can leave a gap.
        boundaries.append(offset + 0.5 * (left_end + right_start))
    return {
        "symbols": symbols,
        "centers_sec": centers,
        "boundaries_sec": boundaries,
        "starts_sec": starts,
        "ends_sec": ends,
        "durations_sec": [end - start for start, end in zip(starts, ends)],
        "edge_start_sec": offset,
        "edge_end_sec": offset + float(supervision.duration),
    }


def gold_word_alignment(cut) -> Dict[str, Any]:
    supervision = cut.supervisions[0]
    if not supervision.alignment or "word" not in supervision.alignment:
        raise ValueError(f"{cut.id}: missing alignment['word']")
    offset = float(supervision.start)
    items = [item for item in supervision.alignment["word"] if item.symbol.strip()]
    starts = [offset + float(item.start) for item in items]
    ends = [
        offset + float(item.start) + float(item.duration) for item in items
    ]
    return {
        "symbols": [item.symbol for item in items],
        "starts_sec": starts,
        "ends_sec": ends,
        "durations_sec": [end - start for start, end in zip(starts, ends)],
    }


def frame_to_seconds(
    frame: float,
    frame_shift: float = 0.01,
    subsampling_factor: int = 4,
) -> float:
    if subsampling_factor != 4:
        raise ValueError("TIMIT alignment timing currently supports subsampling_factor=4")
    # Conv2dSubsampling uses two kernel-3, stride-2 convolutions without padding.
    # Output frame k is centered on input feature frame 3 + 4*k.
    return (3.5 + float(frame) * subsampling_factor) * float(frame_shift)


def greedy_alignment(
    log_probs: torch.Tensor,
    token_table,
    frame_shift: float = 0.01,
    subsampling_factor: int = 4,
) -> Dict[str, Any]:
    runs = greedy_runs(log_probs)
    symbols = token_ids_to_symbols([run["token_id"] for run in runs], token_table)
    centers = [
        frame_to_seconds(
            run["center_frame"],
            frame_shift=frame_shift,
            subsampling_factor=subsampling_factor,
        )
        for run in runs
    ]
    boundaries = [0.5 * (a + b) for a, b in zip(centers[:-1], centers[1:])]
    return {
        "symbols": symbols,
        "centers_sec": centers,
        "boundaries_sec": boundaries,
    }


def gold_frame_token_ids(
    labels: torch.Tensor,
    gold_boundaries_sec: Sequence[float],
    num_frames: int,
    frame_shift: float = 0.01,
    subsampling_factor: int = 4,
) -> torch.Tensor:
    """Assign each encoder frame the GT phone containing its center time."""
    if labels.numel() == 0:
        return labels.new_zeros((num_frames,))
    positions = [
        bisect.bisect_right(
            gold_boundaries_sec,
            frame_to_seconds(
                frame,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
            ),
        )
        for frame in range(num_frames)
    ]
    position_tensor = torch.tensor(positions, dtype=torch.long, device=labels.device)
    return labels[position_tensor.clamp(max=labels.numel() - 1)]


def plan_frame_token_ids(plan: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Map every frame to the transcript token receiving the most plan mass."""
    if plan.size(0) == 0:
        return labels.new_zeros((0,))
    if labels.numel() == 0 or plan.size(1) == 0:
        return labels.new_zeros((plan.size(0),))
    token_positions = plan.argmax(dim=1).to(labels.device)
    return labels[token_positions.clamp(max=labels.numel() - 1)]


def boundary_frame_token_ids(
    labels: torch.Tensor,
    boundaries_sec: Sequence[float],
    num_frames: int,
    frame_shift: float = 0.01,
    subsampling_factor: int = 4,
) -> torch.Tensor:
    """Assign frames using a monotonic transcript alignment's boundaries."""
    return gold_frame_token_ids(
        labels=labels,
        gold_boundaries_sec=boundaries_sec,
        num_frames=num_frames,
        frame_shift=frame_shift,
        subsampling_factor=subsampling_factor,
    )


def add_frame_accuracy(
    row: Dict[str, Any],
    prefix: str,
    predicted_token_ids: torch.Tensor,
    gold_token_ids: torch.Tensor,
) -> None:
    if predicted_token_ids.numel() != gold_token_ids.numel():
        raise ValueError(
            f"{prefix}: predicted/gold frame count mismatch: "
            f"{predicted_token_ids.numel()} vs {gold_token_ids.numel()}"
        )
    count = int(gold_token_ids.numel())
    correct = int((predicted_token_ids == gold_token_ids).sum().item())
    row[f"{prefix}__frame_correct"] = correct
    row[f"{prefix}__frame_count"] = count
    row[f"{prefix}__frame_accuracy"] = correct / max(count, 1)


def isotonic_non_decreasing(
    values: Sequence[float],
    weights: Sequence[float],
) -> List[float]:
    """Weighted PAVA projection onto a non-decreasing sequence."""
    blocks: List[List[float]] = []
    for value, weight in zip(values, weights):
        blocks.append([float(value), max(float(weight), 1.0e-8), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right_value, right_weight, right_count = blocks.pop()
            left_value, left_weight, left_count = blocks.pop()
            merged_weight = left_weight + right_weight
            merged_value = (
                left_value * left_weight + right_value * right_weight
            ) / merged_weight
            blocks.append(
                [merged_value, merged_weight, int(left_count + right_count)]
            )
    projected: List[float] = []
    for value, _, count in blocks:
        projected.extend([float(value)] * int(count))
    return projected


def plan_alignment(
    plan: torch.Tensor,
    frame_shift: float = 0.01,
    subsampling_factor: int = 4,
    enforce_monotonic_centers: bool = False,
    edge_start_sec: float = 0.0,
    edge_end_sec: float = None,
) -> Dict[str, Any]:
    num_frames, num_tokens = plan.shape
    frame_sec = torch.tensor(
        [
            frame_to_seconds(
                frame,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
            )
            for frame in range(num_frames)
        ],
        dtype=plan.dtype,
        device=plan.device,
    )
    col_mass = plan.sum(dim=0).clamp_min(1.0e-8)
    centers = ((plan * frame_sec.unsqueeze(1)).sum(dim=0) / col_mass).tolist()
    if enforce_monotonic_centers:
        centers = isotonic_non_decreasing(centers, col_mass.tolist())
    boundaries = [0.5 * (a + b) for a, b in zip(centers[:-1], centers[1:])]
    if edge_end_sec is None:
        edge_end_sec = (
            frame_to_seconds(
                num_frames - 1,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
            )
            if num_frames > 0
            else edge_start_sec
        )
    starts = [float(edge_start_sec), *boundaries]
    ends = [*boundaries, float(edge_end_sec)]
    return {
        "centers_sec": [float(x) for x in centers],
        "boundaries_sec": [float(x) for x in boundaries],
        "starts_sec": [float(x) for x in starts],
        "ends_sec": [float(x) for x in ends],
        "durations_sec": [float(end - start) for start, end in zip(starts, ends)],
        "num_tokens": num_tokens,
    }


def exact_match_center_errors(
    gold_symbols: Sequence[str],
    gold_centers: Sequence[float],
    pred_symbols: Sequence[str],
    pred_centers: Sequence[float],
) -> List[float]:
    errors = []
    ref_idx = 0
    hyp_idx = 0
    for ref, hyp in kaldialign.align(gold_symbols, pred_symbols, "<eps>"):
        if ref != "<eps>" and hyp != "<eps>" and ref == hyp:
            errors.append(abs(float(gold_centers[ref_idx]) - float(pred_centers[hyp_idx])))
        if ref != "<eps>":
            ref_idx += 1
        if hyp != "<eps>":
            hyp_idx += 1
    return errors


def boundary_match_counts(
    predicted: Sequence[float],
    gold: Sequence[float],
    tolerance_sec: float,
) -> Tuple[int, int, int]:
    predicted = sorted(float(x) for x in predicted)
    gold = sorted(float(x) for x in gold)
    pred_idx = 0
    gold_idx = 0
    matches = 0
    while pred_idx < len(predicted) and gold_idx < len(gold):
        delta = predicted[pred_idx] - gold[gold_idx]
        if abs(delta) <= tolerance_sec:
            matches += 1
            pred_idx += 1
            gold_idx += 1
        elif delta < -tolerance_sec:
            pred_idx += 1
        else:
            gold_idx += 1
    return matches, len(predicted), len(gold)


def nearest_boundary_errors(
    predicted: Sequence[float],
    gold: Sequence[float],
) -> List[float]:
    if not predicted:
        return []
    return [min(abs(float(g) - float(p)) for p in predicted) for g in gold]


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else float("nan")


def add_error_metrics(
    row: Dict[str, Any],
    prefix: str,
    center_errors: Sequence[float],
    boundary_errors: Sequence[float],
) -> None:
    row[f"{prefix}__center_count"] = len(center_errors)
    row[f"{prefix}__center_abs_error_sum_ms"] = 1000.0 * sum(center_errors)
    row[f"{prefix}__center_mae_ms"] = 1000.0 * safe_mean(center_errors)
    row[f"{prefix}__center_p90_ms"] = 1000.0 * safe_percentile(center_errors, 90)
    row[f"{prefix}__boundary_error_count"] = len(boundary_errors)
    row[f"{prefix}__boundary_abs_error_sum_ms"] = 1000.0 * sum(boundary_errors)
    row[f"{prefix}__boundary_nearest_mae_ms"] = 1000.0 * safe_mean(boundary_errors)


def add_boundary_metrics(
    row: Dict[str, Any],
    prefix: str,
    predicted: Sequence[float],
    gold: Sequence[float],
    tolerances_ms: Sequence[float],
) -> None:
    for tolerance_ms in tolerances_ms:
        tag = f"{tolerance_ms:g}ms"
        matches, pred_count, gold_count = boundary_match_counts(
            predicted, gold, tolerance_sec=float(tolerance_ms) / 1000.0
        )
        precision = matches / pred_count if pred_count else 0.0
        recall = matches / gold_count if gold_count else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        row[f"{prefix}__boundary_matches_{tag}"] = matches
        row[f"{prefix}__boundary_pred_count_{tag}"] = pred_count
        row[f"{prefix}__boundary_gold_count_{tag}"] = gold_count
        row[f"{prefix}__boundary_precision_{tag}"] = precision
        row[f"{prefix}__boundary_recall_{tag}"] = recall
        row[f"{prefix}__boundary_f1_{tag}"] = f1


def _interval_error_sums(
    gold_starts: Sequence[float],
    gold_ends: Sequence[float],
    predicted_starts: Sequence[float],
    predicted_ends: Sequence[float],
    indices: Sequence[int] = None,
) -> Dict[str, float]:
    if not (
        len(gold_starts)
        == len(gold_ends)
        == len(predicted_starts)
        == len(predicted_ends)
    ):
        raise ValueError("Gold and predicted interval lengths differ")
    selected = list(range(len(gold_starts))) if indices is None else list(indices)
    onset = [abs(predicted_starts[i] - gold_starts[i]) for i in selected]
    offset = [abs(predicted_ends[i] - gold_ends[i]) for i in selected]
    duration_signed = [
        (predicted_ends[i] - predicted_starts[i])
        - (gold_ends[i] - gold_starts[i])
        for i in selected
    ]
    gold_duration = [gold_ends[i] - gold_starts[i] for i in selected]
    predicted_duration = [
        predicted_ends[i] - predicted_starts[i] for i in selected
    ]
    return {
        "count": len(selected),
        "onset_abs_sum_ms": 1000.0 * sum(onset),
        "offset_abs_sum_ms": 1000.0 * sum(offset),
        "boundary_abs_sum_ms": 500.0 * (sum(onset) + sum(offset)),
        "duration_abs_sum_ms": 1000.0 * sum(abs(x) for x in duration_signed),
        "duration_signed_sum_ms": 1000.0 * sum(duration_signed),
        "gold_duration_sum_ms": 1000.0 * sum(gold_duration),
        "predicted_duration_sum_ms": 1000.0 * sum(predicted_duration),
    }


def _assign_phones_to_words(
    phone_starts: Sequence[float],
    phone_ends: Sequence[float],
    word_starts: Sequence[float],
    word_ends: Sequence[float],
) -> Tuple[List[List[int]], int]:
    """Assign each reference phone index to the word with maximum overlap."""
    groups: List[List[int]] = [[] for _ in word_starts]
    unassigned = 0
    for phone_index, (phone_start, phone_end) in enumerate(
        zip(phone_starts, phone_ends)
    ):
        overlaps = [
            max(0.0, min(phone_end, word_end) - max(phone_start, word_start))
            for word_start, word_end in zip(word_starts, word_ends)
        ]
        if not overlaps or max(overlaps) <= 0.0:
            unassigned += 1
            continue
        groups[int(np.argmax(overlaps))].append(phone_index)
    return groups, unassigned


def add_forced_interval_metrics(
    row: Dict[str, Any],
    prefix: str,
    predicted: Dict[str, Any],
    gold_phone: Dict[str, Any],
    gold_word: Dict[str, Any],
) -> None:
    phone_all = _interval_error_sums(
        gold_phone["starts_sec"],
        gold_phone["ends_sec"],
        predicted["starts_sec"],
        predicted["ends_sec"],
    )
    nonsil_indices = [
        i for i, symbol in enumerate(gold_phone["symbols"]) if symbol != "sil"
    ]
    phone_nonsil = _interval_error_sums(
        gold_phone["starts_sec"],
        gold_phone["ends_sec"],
        predicted["starts_sec"],
        predicted["ends_sec"],
        indices=nonsil_indices,
    )
    for name, metrics in (("phone", phone_all), ("phone_nonsil", phone_nonsil)):
        for metric, value in metrics.items():
            row[f"{prefix}__{name}_{metric}"] = value

    groups, unassigned = _assign_phones_to_words(
        gold_phone["starts_sec"],
        gold_phone["ends_sec"],
        gold_word["starts_sec"],
        gold_word["ends_sec"],
    )
    predicted_word_starts = []
    predicted_word_ends = []
    gold_word_starts = []
    gold_word_ends = []
    for word_index, phone_indices in enumerate(groups):
        if not phone_indices:
            continue
        predicted_word_starts.append(predicted["starts_sec"][phone_indices[0]])
        predicted_word_ends.append(predicted["ends_sec"][phone_indices[-1]])
        gold_word_starts.append(gold_word["starts_sec"][word_index])
        gold_word_ends.append(gold_word["ends_sec"][word_index])
    word = _interval_error_sums(
        gold_word_starts,
        gold_word_ends,
        predicted_word_starts,
        predicted_word_ends,
    )
    for metric, value in word.items():
        row[f"{prefix}__word_{metric}"] = value
    row[f"{prefix}__word_total"] = len(groups)
    row[f"{prefix}__word_mapped"] = word["count"]
    row[f"{prefix}__phone_unassigned_to_word"] = unassigned


def _encoder_aligned_acoustic_features(
    features: torch.Tensor,
    raw_num_frames: int,
    encoder_length: int,
    subsampling_factor: int,
) -> torch.Tensor:
    indices = (
        torch.arange(encoder_length, device=features.device) * subsampling_factor
        + subsampling_factor // 2
    ).clamp(max=max(raw_num_frames - 1, 0))
    return features.index_select(0, indices)


def true_vi_plan(
    output,
    labels: torch.Tensor,
    vi_params,
    gate: str,
    raw_features: torch.Tensor,
    raw_num_frames: int,
    plan_eps: float = None,
    lambda_gw_override: float = None,
) -> torch.Tensor:
    if gate == "posterior":
        alpha = output.alpha_post
    elif gate == "prior":
        alpha = output.alpha_prior
    elif gate == "train":
        if output.alpha_post is None or output.alpha_prior is None:
            raise ValueError("Training gate reconstruction requires both gates")
        prior_mix = min(max(float(getattr(vi_params, "train_prior_mix", 0.0)), 0.0), 1.0)
        alpha = prior_mix * output.alpha_prior + (1.0 - prior_mix) * output.alpha_post
    else:
        raise ValueError(f"Unsupported plan gate: {gate}")
    if alpha is None or output.log_p_nonblank is None:
        raise ValueError("VI output is missing the tensors needed for the OT plan")
    lambda_gw = (
        float(getattr(vi_params, "lambda_gw", 0.0))
        if lambda_gw_override is None
        else float(lambda_gw_override)
    )
    common = dict(
        log_p_nonblank=output.log_p_nonblank,
        alpha=alpha,
        labels=labels,
        column_marginal_type=getattr(vi_params, "col_marginal_type", "acoustic"),
        alpha_smooth_mix=getattr(vi_params, "alpha_smooth_mix", 0.1),
        bpe_col_floor=getattr(vi_params, "bpe_col_floor", 0.05),
        token_prior_sigma=getattr(vi_params, "ot_token_prior_sigma", 0.15),
        token_prior_score_temp=getattr(vi_params, "ot_token_prior_score_temp", 1.0),
        token_prior_floor=getattr(vi_params, "ot_token_prior_floor", 0.05),
        eps=(plan_eps if plan_eps is not None else getattr(vi_params, "ot_eps", 0.3)),
        iters=getattr(vi_params, "ot_iters", 10),
        beta_pos=getattr(vi_params, "ot_beta_pos", 1.0),
        return_plan=True,
    )
    if lambda_gw > 0.0:
        acoustic = _encoder_aligned_acoustic_features(
            raw_features,
            raw_num_frames=raw_num_frames,
            encoder_length=output.log_p_nonblank.size(0),
            subsampling_factor=int(getattr(vi_params, "subsampling_factor", 4)),
        )
        _, plan = vi_fgw_loss_v2(
            **common,
            acoustic_features=acoustic,
            lambda_gw=lambda_gw,
            n_outer=int(getattr(vi_params, "gw_n_outer", 3)),
        )
    else:
        _, plan = vi_ot_loss_v2(**common)
    if plan is None:
        raise ValueError("VI OT/FGW plan is empty")
    return plan.detach().cpu()


def alignment_geometry(plan: torch.Tensor) -> Dict[str, float]:
    plan = plan.float()
    num_frames, num_tokens = plan.shape
    t_pos = torch.linspace(0, 1, num_frames).unsqueeze(1)
    u_pos = torch.linspace(0, 1, num_tokens).unsqueeze(0)
    distance = (t_pos - u_pos).abs()
    mass = plan.sum().clamp_min(1.0e-8)
    peaks = plan.max(dim=0).values
    support = (plan >= 0.1 * peaks.unsqueeze(0)) & (peaks > 0).unsqueeze(0)
    return {
        "diag_mean_abs_dev": float((plan * distance).sum().div(mass)),
        "offdiag_mass": float(plan[distance > 0.12].sum().div(mass)),
        "support_mean_frames": float(support.sum(dim=0).float().mean()),
    }


def plan_agreement(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    values = compute_plan_agreement_metrics(
        left, right, support_relative_threshold=0.1
    )
    return {
        "barycenter_mad": values["plan_ctc_barycenter_mad"],
        "support_iou": values["plan_ctc_support_iou"],
        "total_variation": values["plan_ctc_total_variation"],
    }


def ctc_forced_plan(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return monotonic CTC token-state posterior occupancy for one utterance."""
    plan = ctc_token_occupancy_batched(
        log_probs=log_probs.unsqueeze(0),
        labels=labels.unsqueeze(0),
        frame_lens=torch.tensor([log_probs.size(0)], dtype=torch.long),
        label_lens=torch.tensor([labels.numel()], dtype=torch.long),
        blank_id=0,
    )
    return plan[0].detach().cpu()


def collect_split_rows(
    dl,
    baseline_model,
    vi_model,
    graph,
    vi_params,
    device,
    args,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for batch_idx, batch in enumerate(dl):
        baseline_outputs = baseline_batch_outputs(baseline_model, batch, device)
        vi_outputs, supervision_ids = vi_batch_outputs(
            vi_model,
            batch,
            graph=graph,
            device=device,
            gate=("prior" if args.vi_plan_gate == "train" else args.vi_plan_gate),
        )
        sequence_idx = batch["supervisions"]["sequence_idx"].tolist()
        cuts = batch["supervisions"]["cut"]

        for sup_idx, cut in enumerate(cuts):
            seq_idx = int(sequence_idx[sup_idx])
            labels = torch.tensor(supervision_ids[sup_idx], dtype=torch.long)
            pieces = [graph.token_table[int(token_id)] for token_id in labels.tolist()]
            gold = gold_phone_alignment(cut)
            gold_word = gold_word_alignment(cut)
            if pieces != gold["symbols"]:
                raise ValueError(f"{cut.id}: graph labels do not match gold phones")

            baseline_output = baseline_outputs[seq_idx]
            vi_output = vi_outputs[seq_idx]
            frame_shift = float(getattr(cut.features, "frame_shift", 0.01))
            subsampling_factor = int(getattr(vi_params, "subsampling_factor", 4))
            baseline_greedy = greedy_alignment(
                baseline_output.log_probs,
                token_table=graph.token_table,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
            )
            vi_greedy = greedy_alignment(
                vi_output.log_probs,
                token_table=graph.token_table,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
            )
            baseline_ctc_plan = ctc_forced_plan(baseline_output.log_probs, labels)
            vi_ctc_plan = ctc_forced_plan(vi_output.log_probs, labels)
            baseline_ctc_forced = plan_alignment(
                baseline_ctc_plan,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
                edge_start_sec=gold["edge_start_sec"],
                edge_end_sec=gold["edge_end_sec"],
            )
            vi_ctc_forced = plan_alignment(
                vi_ctc_plan,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
                edge_start_sec=gold["edge_start_sec"],
                edge_end_sec=gold["edge_end_sec"],
            )
            raw_features = batch["inputs"][seq_idx]
            raw_num_frames = int(batch["supervisions"]["num_frames"][sup_idx])
            plan = true_vi_plan(
                vi_output, labels=labels, vi_params=vi_params, gate=args.vi_plan_gate,
                raw_features=raw_features,
                raw_num_frames=raw_num_frames,
                plan_eps=args.plan_ot_eps,
            )
            counterfactual_ot_plan = true_vi_plan(
                vi_output,
                labels=labels,
                vi_params=vi_params,
                gate=args.vi_plan_gate,
                raw_features=raw_features,
                raw_num_frames=raw_num_frames,
                plan_eps=args.plan_ot_eps,
                lambda_gw_override=0.0,
            )
            vi_plan = plan_alignment(
                plan,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
                edge_start_sec=gold["edge_start_sec"],
                edge_end_sec=gold["edge_end_sec"],
            )
            vi_plan_monotonic = plan_alignment(
                plan,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
                enforce_monotonic_centers=True,
                edge_start_sec=gold["edge_start_sec"],
                edge_end_sec=gold["edge_end_sec"],
            )

            baseline_center_errors = exact_match_center_errors(
                gold["symbols"],
                gold["centers_sec"],
                baseline_greedy["symbols"],
                baseline_greedy["centers_sec"],
            )
            vi_greedy_center_errors = exact_match_center_errors(
                gold["symbols"],
                gold["centers_sec"],
                vi_greedy["symbols"],
                vi_greedy["centers_sec"],
            )
            baseline_ctc_center_errors = [
                abs(gold_center - pred_center)
                for gold_center, pred_center in zip(
                    gold["centers_sec"], baseline_ctc_forced["centers_sec"]
                )
            ]
            vi_ctc_center_errors = [
                abs(gold_center - pred_center)
                for gold_center, pred_center in zip(
                    gold["centers_sec"], vi_ctc_forced["centers_sec"]
                )
            ]
            vi_center_errors = [
                abs(gold_center - pred_center)
                for gold_center, pred_center in zip(
                    gold["centers_sec"], vi_plan["centers_sec"]
                )
            ]
            baseline_boundary_errors = nearest_boundary_errors(
                baseline_greedy["boundaries_sec"], gold["boundaries_sec"]
            )
            vi_greedy_boundary_errors = nearest_boundary_errors(
                vi_greedy["boundaries_sec"], gold["boundaries_sec"]
            )
            baseline_ctc_boundary_errors = [
                abs(gold_boundary - pred_boundary)
                for gold_boundary, pred_boundary in zip(
                    gold["boundaries_sec"], baseline_ctc_forced["boundaries_sec"]
                )
            ]
            vi_ctc_boundary_errors = [
                abs(gold_boundary - pred_boundary)
                for gold_boundary, pred_boundary in zip(
                    gold["boundaries_sec"], vi_ctc_forced["boundaries_sec"]
                )
            ]
            vi_boundary_errors = [
                abs(gold_boundary - pred_boundary)
                for gold_boundary, pred_boundary in zip(
                    gold["boundaries_sec"], vi_plan["boundaries_sec"]
                )
            ]
            vi_monotonic_center_errors = [
                abs(gold_center - pred_center)
                for gold_center, pred_center in zip(
                    gold["centers_sec"], vi_plan_monotonic["centers_sec"]
                )
            ]
            vi_monotonic_boundary_errors = [
                abs(gold_boundary - pred_boundary)
                for gold_boundary, pred_boundary in zip(
                    gold["boundaries_sec"], vi_plan_monotonic["boundaries_sec"]
                )
            ]

            baseline_stats = compute_alignment_stats(
                log_probs=baseline_output.log_probs,
                labels=labels,
                token_pieces=pieces,
                blank_id=0,
                tau=args.shared_ot_tau,
                eps=args.shared_ot_eps,
                iters=args.shared_ot_iters,
                beta_pos=args.shared_ot_beta_pos,
                support_relative_threshold=args.support_relative_threshold,
            )
            vi_stats = compute_alignment_stats(
                log_probs=vi_output.log_probs,
                labels=labels,
                token_pieces=pieces,
                blank_id=0,
                tau=args.shared_ot_tau,
                eps=args.shared_ot_eps,
                iters=args.shared_ot_iters,
                beta_pos=args.shared_ot_beta_pos,
                support_relative_threshold=args.support_relative_threshold,
            )
            baseline_shared = compute_alignment_quality_metrics(
                baseline_stats,
                diagonal_band_width=args.diagonal_band_width,
                backward_tol=args.backward_tol,
            )
            vi_shared = compute_alignment_quality_metrics(
                vi_stats,
                diagonal_band_width=args.diagonal_band_width,
                backward_tol=args.backward_tol,
            )

            row: Dict[str, Any] = {
                "cut_id": cut.id,
                "duration_sec": float(cut.duration),
                "num_gold_phones": len(gold["symbols"]),
                "num_baseline_greedy_phones": len(baseline_greedy["symbols"]),
                "baseline_greedy__token_match_count": len(baseline_center_errors),
                "baseline_greedy__token_match_rate": len(baseline_center_errors)
                / max(len(gold["symbols"]), 1),
                "num_vi_greedy_phones": len(vi_greedy["symbols"]),
                "vi_greedy__token_match_count": len(vi_greedy_center_errors),
                "vi_greedy__token_match_rate": len(vi_greedy_center_errors)
                / max(len(gold["symbols"]), 1),
            }
            baseline_gold_frame_ids = gold_frame_token_ids(
                labels=labels,
                gold_boundaries_sec=gold["boundaries_sec"],
                num_frames=baseline_output.output_len,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
            )
            vi_gold_frame_ids = gold_frame_token_ids(
                labels=labels,
                gold_boundaries_sec=gold["boundaries_sec"],
                num_frames=vi_output.output_len,
                frame_shift=frame_shift,
                subsampling_factor=subsampling_factor,
            )
            add_frame_accuracy(
                row,
                "baseline_greedy",
                baseline_output.log_probs.argmax(dim=-1),
                baseline_gold_frame_ids,
            )
            add_frame_accuracy(
                row,
                "baseline_ctc_forced",
                plan_frame_token_ids(baseline_ctc_plan, labels),
                baseline_gold_frame_ids,
            )
            add_frame_accuracy(
                row,
                "vi_greedy",
                vi_output.log_probs.argmax(dim=-1),
                vi_gold_frame_ids,
            )
            add_frame_accuracy(
                row,
                "vi_ctc_forced",
                plan_frame_token_ids(vi_ctc_plan, labels),
                vi_gold_frame_ids,
            )
            add_frame_accuracy(
                row,
                "vi_ot_plan",
                plan_frame_token_ids(plan, labels),
                vi_gold_frame_ids,
            )
            add_frame_accuracy(
                row,
                "vi_ot_plan_monotonic",
                boundary_frame_token_ids(
                    labels=labels,
                    boundaries_sec=vi_plan_monotonic["boundaries_sec"],
                    num_frames=vi_output.output_len,
                    frame_shift=frame_shift,
                    subsampling_factor=subsampling_factor,
                ),
                vi_gold_frame_ids,
            )
            add_error_metrics(
                row,
                "baseline_greedy",
                center_errors=baseline_center_errors,
                boundary_errors=baseline_boundary_errors,
            )
            add_error_metrics(
                row,
                "vi_greedy",
                center_errors=vi_greedy_center_errors,
                boundary_errors=vi_greedy_boundary_errors,
            )
            add_error_metrics(
                row,
                "baseline_ctc_forced",
                center_errors=baseline_ctc_center_errors,
                boundary_errors=baseline_ctc_boundary_errors,
            )
            add_error_metrics(
                row,
                "vi_ctc_forced",
                center_errors=vi_ctc_center_errors,
                boundary_errors=vi_ctc_boundary_errors,
            )
            add_error_metrics(
                row,
                "vi_ot_plan",
                center_errors=vi_center_errors,
                boundary_errors=vi_boundary_errors,
            )
            add_error_metrics(
                row,
                "vi_ot_plan_monotonic",
                center_errors=vi_monotonic_center_errors,
                boundary_errors=vi_monotonic_boundary_errors,
            )
            add_boundary_metrics(
                row,
                "baseline_greedy",
                baseline_greedy["boundaries_sec"],
                gold["boundaries_sec"],
                args.boundary_tolerances_ms,
            )
            add_boundary_metrics(
                row,
                "baseline_ctc_forced",
                baseline_ctc_forced["boundaries_sec"],
                gold["boundaries_sec"],
                args.boundary_tolerances_ms,
            )
            add_boundary_metrics(
                row,
                "vi_ctc_forced",
                vi_ctc_forced["boundaries_sec"],
                gold["boundaries_sec"],
                args.boundary_tolerances_ms,
            )
            add_boundary_metrics(
                row,
                "vi_greedy",
                vi_greedy["boundaries_sec"],
                gold["boundaries_sec"],
                args.boundary_tolerances_ms,
            )
            add_boundary_metrics(
                row,
                "vi_ot_plan",
                vi_plan["boundaries_sec"],
                gold["boundaries_sec"],
                args.boundary_tolerances_ms,
            )
            add_boundary_metrics(
                row,
                "vi_ot_plan_monotonic",
                vi_plan_monotonic["boundaries_sec"],
                gold["boundaries_sec"],
                args.boundary_tolerances_ms,
            )
            for prefix, predicted in (
                ("baseline_ctc_forced", baseline_ctc_forced),
                ("vi_ctc_forced", vi_ctc_forced),
                ("vi_ot_plan", vi_plan),
                ("vi_ot_plan_monotonic", vi_plan_monotonic),
            ):
                add_forced_interval_metrics(
                    row,
                    prefix,
                    predicted=predicted,
                    gold_phone=gold,
                    gold_word=gold_word,
                )

            for prefix, alignment in (
                ("matched_plan_geometry", plan),
                ("counterfactual_ot_geometry", counterfactual_ot_plan),
                ("ctc_geometry", vi_ctc_plan),
            ):
                for key, value in alignment_geometry(alignment).items():
                    row[f"{prefix}__{key}"] = value
            for prefix, values in (
                ("matched_plan_vs_ot", plan_agreement(plan, counterfactual_ot_plan)),
                ("matched_plan_vs_ctc", plan_agreement(plan, vi_ctc_plan)),
                ("counterfactual_ot_vs_ctc", plan_agreement(counterfactual_ot_plan, vi_ctc_plan)),
            ):
                for key, value in values.items():
                    row[f"{prefix}__{key}"] = value
            row["matched_plan_minus_ot_diag_mean_abs_dev"] = (
                row["matched_plan_geometry__diag_mean_abs_dev"]
                - row["counterfactual_ot_geometry__diag_mean_abs_dev"]
            )
            for key, value in baseline_shared.items():
                row[f"baseline_shared__{key}"] = value
            for key, value in vi_shared.items():
                row[f"vi_shared__{key}"] = value
            rows.append(row)
            details.append(
                {
                    "cut_id": cut.id,
                    "gold": gold,
                    "gold_word": gold_word,
                    "baseline_greedy": baseline_greedy,
                    "baseline_ctc_forced": baseline_ctc_forced,
                    "vi_greedy": vi_greedy,
                    "vi_ctc_forced": vi_ctc_forced,
                    "vi_ot_plan": vi_plan,
                    "vi_ot_plan_monotonic": vi_plan_monotonic,
                }
            )
        if batch_idx % 20 == 0:
            logging.info("alignment batch %s", batch_idx)
    return rows, details


def aggregate_method(
    rows: Sequence[Dict[str, Any]],
    prefix: str,
    tolerances_ms: Sequence[float],
) -> Dict[str, Any]:
    center_count = sum(int(row[f"{prefix}__center_count"]) for row in rows)
    boundary_error_count = sum(int(row[f"{prefix}__boundary_error_count"]) for row in rows)
    result: Dict[str, Any] = {
        "center_count": center_count,
        "center_mae_ms": sum(row[f"{prefix}__center_abs_error_sum_ms"] for row in rows)
        / max(center_count, 1),
        "boundary_error_count": boundary_error_count,
        "boundary_nearest_mae_ms": sum(
            row[f"{prefix}__boundary_abs_error_sum_ms"] for row in rows
        )
        / max(boundary_error_count, 1),
        "boundary": {},
    }
    if f"{prefix}__frame_count" in rows[0]:
        frame_count = sum(int(row[f"{prefix}__frame_count"]) for row in rows)
        frame_correct = sum(int(row[f"{prefix}__frame_correct"]) for row in rows)
        result.update(
            {
                "frame_correct": frame_correct,
                "frame_count": frame_count,
                "frame_accuracy": frame_correct / max(frame_count, 1),
            }
        )
    if f"{prefix}__phone_count" in rows[0]:
        interval_results = {}
        for level in ("phone", "phone_nonsil", "word"):
            count_key = f"{prefix}__{level}_count"
            count = sum(int(row[count_key]) for row in rows)
            values: Dict[str, Any] = {"count": count}
            for metric, output_name in (
                ("onset_abs_sum_ms", "onset_mae_ms"),
                ("offset_abs_sum_ms", "offset_mae_ms"),
                ("boundary_abs_sum_ms", "pbe_or_wbe_ms"),
                ("duration_abs_sum_ms", "duration_mae_ms"),
                ("duration_signed_sum_ms", "duration_bias_ms"),
                ("gold_duration_sum_ms", "gold_duration_mean_ms"),
                ("predicted_duration_sum_ms", "predicted_duration_mean_ms"),
            ):
                key = f"{prefix}__{level}_{metric}"
                total = sum(float(row[key]) for row in rows)
                values[output_name] = total / max(count, 1)
                macro_values = [
                    float(row[key]) / int(row[count_key])
                    for row in rows
                    if int(row[count_key]) > 0
                ]
                values[f"macro_{output_name}"] = safe_mean(macro_values)
            interval_results[level] = values
        result["intervals"] = interval_results
    for tolerance_ms in tolerances_ms:
        tag = f"{tolerance_ms:g}ms"
        matches = sum(int(row[f"{prefix}__boundary_matches_{tag}"]) for row in rows)
        pred_count = sum(int(row[f"{prefix}__boundary_pred_count_{tag}"]) for row in rows)
        gold_count = sum(int(row[f"{prefix}__boundary_gold_count_{tag}"]) for row in rows)
        precision = matches / max(pred_count, 1)
        recall = matches / max(gold_count, 1)
        result["boundary"][tag] = {
            "matches": matches,
            "pred_count": pred_count,
            "gold_count": gold_count,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-8),
        }
    return result


def aggregate_shared(rows: Sequence[Dict[str, Any]], prefix: str) -> Dict[str, float]:
    keys = [key for key in rows[0] if key.startswith(f"{prefix}__")]
    return {
        key.split("__", 1)[1]: float(
            np.mean([float(row[key]) for row in rows if math.isfinite(float(row[key]))])
        )
        for key in keys
    }


def aggregate_summary(rows: Sequence[Dict[str, Any]], split: str, args) -> Dict[str, Any]:
    baseline = aggregate_method(rows, "baseline_greedy", args.boundary_tolerances_ms)
    baseline["token_match_rate"] = sum(
        int(row["baseline_greedy__token_match_count"]) for row in rows
    ) / max(sum(int(row["num_gold_phones"]) for row in rows), 1)
    vi_greedy = aggregate_method(rows, "vi_greedy", args.boundary_tolerances_ms)
    vi_greedy["token_match_rate"] = sum(
        int(row["vi_greedy__token_match_count"]) for row in rows
    ) / max(sum(int(row["num_gold_phones"]) for row in rows), 1)
    baseline_ctc_forced = aggregate_method(
        rows, "baseline_ctc_forced", args.boundary_tolerances_ms
    )
    vi_ctc_forced = aggregate_method(
        rows, "vi_ctc_forced", args.boundary_tolerances_ms
    )
    vi = aggregate_method(rows, "vi_ot_plan", args.boundary_tolerances_ms)
    vi_monotonic = aggregate_method(
        rows, "vi_ot_plan_monotonic", args.boundary_tolerances_ms
    )
    comparison = {
        "vi_greedy_minus_baseline_center_mae_ms": vi_greedy["center_mae_ms"]
        - baseline["center_mae_ms"],
        "vi_greedy_minus_baseline_boundary_nearest_mae_ms": vi_greedy[
            "boundary_nearest_mae_ms"
        ]
        - baseline["boundary_nearest_mae_ms"],
        "vi_greedy_minus_baseline_boundary_f1": {
            tag: vi_greedy["boundary"][tag]["f1"] - baseline["boundary"][tag]["f1"]
            for tag in baseline["boundary"]
        },
        "vi_ctc_forced_minus_baseline_ctc_forced_frame_accuracy": vi_ctc_forced[
            "frame_accuracy"
        ]
        - baseline_ctc_forced["frame_accuracy"],
        "vi_ctc_forced_minus_baseline_ctc_forced_boundary_nearest_mae_ms": (
            vi_ctc_forced["boundary_nearest_mae_ms"]
            - baseline_ctc_forced["boundary_nearest_mae_ms"]
        ),
        "vi_ctc_forced_minus_baseline_ctc_forced_boundary_f1": {
            tag: vi_ctc_forced["boundary"][tag]["f1"]
            - baseline_ctc_forced["boundary"][tag]["f1"]
            for tag in baseline_ctc_forced["boundary"]
        },
        "vi_minus_baseline_center_mae_ms": vi["center_mae_ms"]
        - baseline["center_mae_ms"],
        "vi_minus_baseline_boundary_nearest_mae_ms": vi["boundary_nearest_mae_ms"]
        - baseline["boundary_nearest_mae_ms"],
        "vi_minus_baseline_boundary_f1": {
            tag: vi["boundary"][tag]["f1"] - baseline["boundary"][tag]["f1"]
            for tag in baseline["boundary"]
        },
        "vi_monotonic_minus_baseline_center_mae_ms": vi_monotonic["center_mae_ms"]
        - baseline["center_mae_ms"],
        "vi_monotonic_minus_baseline_boundary_nearest_mae_ms": vi_monotonic[
            "boundary_nearest_mae_ms"
        ]
        - baseline["boundary_nearest_mae_ms"],
        "vi_monotonic_minus_baseline_boundary_f1": {
            tag: vi_monotonic["boundary"][tag]["f1"]
            - baseline["boundary"][tag]["f1"]
            for tag in baseline["boundary"]
        },
    }
    return {
        "split": split,
        "num_utterances": len(rows),
        "vi_plan_gate": args.vi_plan_gate,
        "lambda_gw": float(getattr(args, "checkpoint_lambda_gw", 0.0)),
        "frame_accuracy_definition": (
            "Phone-label accuracy at subsampled encoder-frame centers. Greedy uses "
            "CTC argmax (blank counts as incorrect); CTC-forced and raw OT methods "
            "use the highest-mass transcript token at each frame; monotonic OT uses "
            "the intervals induced by its projected token centers."
        ),
        "forced_alignment_definition": (
            "Transcript-conditioned CTC forward-backward posterior occupancy. Token "
            "centers are posterior means and boundaries are adjacent-center midpoints; "
            "this is posterior forced alignment, not a Viterbi path."
        ),
        "baseline_greedy": baseline,
        "baseline_ctc_forced": baseline_ctc_forced,
        "vi_greedy": vi_greedy,
        "vi_ctc_forced": vi_ctc_forced,
        "vi_ot_plan": vi,
        "vi_ot_plan_monotonic": vi_monotonic,
        "comparison": comparison,
        "shared_alignment_quality": {
            "baseline": aggregate_shared(rows, "baseline_shared"),
            "vi": aggregate_shared(rows, "vi_shared"),
        },
        "fgw_geometry": {
            "matched_plan": aggregate_shared(rows, "matched_plan_geometry"),
            "counterfactual_ot": aggregate_shared(
                rows, "counterfactual_ot_geometry"
            ),
            "ctc": aggregate_shared(rows, "ctc_geometry"),
            "matched_plan_vs_ot": aggregate_shared(
                rows, "matched_plan_vs_ot"
            ),
            "matched_plan_vs_ctc": aggregate_shared(
                rows, "matched_plan_vs_ctc"
            ),
            "counterfactual_ot_vs_ctc": aggregate_shared(
                rows, "counterfactual_ot_vs_ctc"
            ),
            "matched_plan_minus_ot_diag_mean_abs_dev": safe_mean(
                [row["matched_plan_minus_ot_diag_mean_abs_dev"] for row in rows]
            ),
        },
    }


def write_outputs(
    rows: Sequence[Dict[str, Any]],
    details: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    split: str,
    output_dir: Path,
) -> None:
    csv_path = output_dir / f"{split}-alignment-metrics.csv"
    jsonl_path = output_dir / f"{split}-alignment-details.jsonl"
    summary_path = output_dir / f"{split}-alignment-summary.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(jsonl_path, "w") as f:
        for detail in details:
            f.write(json.dumps(detail, sort_keys=True) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logging.info("saved %s", summary_path)


def log_summary(summary: Dict[str, Any]) -> None:
    logging.info("alignment summary: %s (%s utterances)", summary["split"], summary["num_utterances"])
    for method in (
        "baseline_greedy",
        "baseline_ctc_forced",
        "vi_greedy",
        "vi_ctc_forced",
        "vi_ot_plan",
        "vi_ot_plan_monotonic",
    ):
        values = summary[method]
        logging.info(
            "%s center MAE %.2f ms | boundary nearest MAE %.2f ms | frame acc %.4f",
            method,
            values["center_mae_ms"],
            values["boundary_nearest_mae_ms"],
            values["frame_accuracy"],
        )
        for tag, metric in values["boundary"].items():
            logging.info(
                "%s boundary %s: P %.4f R %.4f F1 %.4f",
                method,
                tag,
                metric["precision"],
                metric["recall"],
                metric["f1"],
            )


def main() -> None:
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(args.output_dir / "log-evaluate-timit-alignment"))
    device = resolve_device(args.device)
    logging.info("device: %s", device)

    _, graph = build_phone_graph(args.lang_dir, device=device)
    baseline_model, _ = build_baseline_model(
        args.baseline_checkpoint, args.lang_dir, device=device
    )
    vi_model, vi_params = build_vi_model(
        args.vi_checkpoint,
        args.lang_dir,
        device=device,
        prior_logit_bias=args.vi_prior_logit_bias,
    )
    args.checkpoint_lambda_gw = float(getattr(vi_params, "lambda_gw", 0.0))
    if args.shared_ot_eps is None:
        args.shared_ot_eps = float(getattr(vi_params, "ot_eps", 0.3))
    if args.shared_ot_iters is None:
        args.shared_ot_iters = int(getattr(vi_params, "ot_iters", 10))
    if args.shared_ot_beta_pos is None:
        args.shared_ot_beta_pos = float(getattr(vi_params, "ot_beta_pos", 1.0))

    combined_summary = {}
    for split in args.splits:
        rows, details = collect_split_rows(
            get_split_dataloader(args, split),
            baseline_model=baseline_model,
            vi_model=vi_model,
            graph=graph,
            vi_params=vi_params,
            device=device,
            args=args,
        )
        if not rows:
            raise ValueError(f"No rows collected for {split}")
        summary = aggregate_summary(rows, split=split, args=args)
        write_outputs(rows, details, summary, split=split, output_dir=args.output_dir)
        log_summary(summary)
        combined_summary[split] = summary

    with open(args.output_dir / "alignment-summary.json", "w") as f:
        json.dump(combined_summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
