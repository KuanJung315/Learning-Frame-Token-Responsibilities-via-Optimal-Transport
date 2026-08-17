#!/usr/bin/env python3
"""Train conformer_ctc2 VFTA/VI with an FGW alignment prior."""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_ASR_DIR = _RECIPE_DIR.parent
for _path in (_RECIPE_DIR, _ASR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_vi_ot_v2 as ours_train  # noqa: E402
from ctc_occupancy import fb_posterior_consistency_loss  # noqa: E402
from ctc_plan_consistency import (  # noqa: E402
    ctc_token_occupancy_batched,
    plan_w1_loss_for_gates,
)
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler  # noqa: E402
from icefall.graph_compiler import CtcTrainingGraphCompiler  # noqa: E402
from icefall.utils import (  # noqa: E402
    AttributeDict,
    MetricsTracker,
    encode_supervisions,
    str2bool,
)
from ot_fgw import (  # noqa: E402
    DiagonalResidualPSDFrameMetric,
    LearnablePSDFrameMetric,
    metric_occupancy_geometry_loss,
    vi_fgw_loss_v2,
)
from varctc_v2_utils import (  # noqa: E402
    build_gated_log_probs_v2,
    mean_valid_abs_diff,
    mean_valid_frame_std,
)
from word_phone_graph_compiler import (  # noqa: E402
    WordPhoneCtcTrainingGraphCompiler,
)


GraphCompiler = Union[
    BpeCtcTrainingGraphCompiler,
    CtcTrainingGraphCompiler,
    WordPhoneCtcTrainingGraphCompiler,
]
_BASE_COMPUTE_LOSS = ours_train.compute_loss
_BASE_SAVE_CHECKPOINT = ours_train.save_checkpoint


def _parse_retained_epochs(value: str) -> set[int]:
    value = value.strip()
    if not value:
        return set()
    epochs = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(epoch < 0 for epoch in epochs):
        raise ValueError("retained epoch checkpoints must be non-negative")
    return epochs


def _prune_epoch_checkpoints(params: AttributeDict, rank: int = 0) -> None:
    """Bound disk use while preserving resume and interval-average anchors."""
    if rank != 0:
        return
    keep_last = int(getattr(params, "keep_last_epoch_checkpoints", 0))
    if keep_last <= 0:
        return

    checkpoints = []
    for path in Path(params.exp_dir).glob("epoch-*.pt"):
        try:
            epoch = int(path.stem.removeprefix("epoch-"))
        except ValueError:
            continue
        checkpoints.append((epoch, path))
    checkpoints.sort()

    retained = _parse_retained_epochs(
        str(getattr(params, "retain_epoch_checkpoints", ""))
    )
    retained.update(epoch for epoch, _ in checkpoints[-keep_last:])
    for epoch, path in checkpoints:
        if epoch not in retained:
            path.unlink()
            logging.info("Removed pruned epoch checkpoint %s", path)


def save_checkpoint(*args, **kwargs) -> None:
    """Delegate checkpoint creation, then apply opt-in disk retention."""
    _BASE_SAVE_CHECKPOINT(*args, **kwargs)
    params = kwargs.get("params", args[0] if args else None)
    rank = int(kwargs.get("rank", 0))
    if params is None or rank != 0:
        return
    if not bool(getattr(params, "save_best_checkpoints", True)):
        for name in ("best-train-loss.pt", "best-valid-loss.pt"):
            path = Path(params.exp_dir) / name
            if path.exists():
                path.unlink()
                logging.info("Removed disabled best-loss checkpoint %s", path)
    _prune_epoch_checkpoints(params=params, rank=rank)


def get_parser() -> argparse.ArgumentParser:
    parser = ours_train.get_parser()
    parser.set_defaults(exp_dir="conformer_ctc2/vfta_fgw/exp_li100h")
    parser.add_argument(
        "--keep-last-epoch-checkpoints",
        type=int,
        default=0,
        help=(
            "If positive, retain only this many newest epoch checkpoints plus "
            "the anchors in --retain-epoch-checkpoints. Zero keeps all epochs."
        ),
    )
    parser.add_argument(
        "--retain-epoch-checkpoints",
        type=str,
        default="",
        help="Comma-separated epoch checkpoints retained as averaging anchors.",
    )
    parser.add_argument(
        "--save-best-checkpoints",
        type=str2bool,
        default=True,
        help="Whether to retain raw best-train/best-valid checkpoint copies.",
    )
    parser.add_argument(
        "--lambda-gw",
        type=float,
        default=0.1,
        help=(
            "FGW structural weight in the detached transport plan. "
            "Set 0 to recover the original batched VFTA OT recipe."
        ),
    )
    parser.add_argument(
        "--gw-n-outer",
        type=int,
        default=3,
        help="Frank-Wolfe outer iterations used to update the FGW plan.",
    )
    parser.add_argument(
        "--lambda-gw-warmup-start",
        type=int,
        default=0,
        help=(
            "Global train batch index at which the FGW structural weight "
            "starts ramping from zero. Default keeps the original constant "
            "lambda_gw behavior."
        ),
    )
    parser.add_argument(
        "--lambda-gw-warmup-steps",
        type=int,
        default=0,
        help=(
            "Number of global train batches used to linearly ramp lambda_gw "
            "to its target value after lambda_gw_warmup_start."
        ),
    )
    parser.add_argument(
        "--frame-metric",
        choices=["fixed-cosine", "learned-psd", "learned-diag-psd"],
        default="fixed-cosine",
        help="Frame relation used by the FGW structural term.",
    )
    parser.add_argument(
        "--metric-rho",
        type=float,
        default=1.0,
        help="Interpolation from fixed cosine (0) to learned PSD cost (1).",
    )
    parser.add_argument(
        "--metric-max-log-scale",
        type=float,
        default=0.5,
        help=(
            "Maximum absolute log scale of each Mel dimension for "
            "--frame-metric=learned-diag-psd."
        ),
    )
    parser.add_argument(
        "--metric-normalization",
        choices=["none", "offdiag-rms"],
        default="none",
        help="Optional scale normalization for the learned frame cost.",
    )
    parser.add_argument(
        "--metric-grad-scale",
        type=float,
        default=1.0,
        help="Selective plan-to-metric gradient scale.",
    )
    parser.add_argument(
        "--metric-surrogate",
        choices=["token-cost", "ctc-occupancy"],
        default="token-cost",
        help=(
            "Target for the selective metric gradient. ctc-occupancy uses "
            "blank-aware prior-gate forward-backward occupancy; token-cost "
            "retains the original negative ablation."
        ),
    )
    parser.add_argument(
        "--metric-occ-w1-weight",
        type=float,
        default=1.0,
        help="Normalized temporal W1 weight in the metric-side occupancy loss.",
    )
    parser.add_argument(
        "--metric-occ-barycenter-weight",
        type=float,
        default=0.5,
        help="Normalized barycenter-error weight in the metric-side loss.",
    )
    parser.add_argument(
        "--metric-occ-log-std-weight",
        type=float,
        default=0.1,
        help="Log occupancy-width error weight in the metric-side loss.",
    )
    parser.add_argument(
        "--metric-lr-scale",
        type=float,
        default=0.1,
        help="Metric optimizer learning rate relative to the main model.",
    )
    parser.add_argument(
        "--metric-warmup-start",
        type=int,
        default=1000,
        help="Global batch where the selective metric gradient starts.",
    )
    parser.add_argument(
        "--metric-warmup-steps",
        type=int,
        default=2000,
        help="Batches used to ramp the selective metric gradient to its target.",
    )
    parser.add_argument(
        "--metric-moment-reg-weight",
        type=float,
        default=0.1,
        help="Match learned distance moments to the fixed cosine metric.",
    )
    parser.add_argument(
        "--metric-identity-reg-weight",
        type=float,
        default=0.01,
        help="Regularize L^T L toward identity to prevent metric collapse.",
    )
    parser.add_argument(
        "--metric-spectrum-reg-weight",
        type=float,
        default=0.0,
        help="Log-determinant plus upper-eigenvalue PSD barrier weight.",
    )
    parser.add_argument(
        "--metric-spectrum-max-eigenvalue",
        type=float,
        default=4.0,
        help="Upper Gram eigenvalue allowed before the spectrum hinge activates.",
    )
    return parser


def _configure_vfta_model(
    params: AttributeDict, model: nn.Module
) -> nn.Module:
    frame_metric = str(getattr(params, "frame_metric", "fixed-cosine"))
    if frame_metric == "learned-psd":
        model.structural_metric = LearnablePSDFrameMetric(
            feature_dim=int(params.feature_dim)
        )
    elif frame_metric == "learned-diag-psd":
        model.structural_metric = DiagonalResidualPSDFrameMetric(
            feature_dim=int(params.feature_dim),
            max_log_scale=float(params.metric_max_log_scale),
        )
    return model


def _effective_lambda_gw(params: AttributeDict) -> float:
    return ours_train._linear_ramp(
        batch_idx=int(getattr(params, "batch_idx_train", 0)),
        start=int(getattr(params, "lambda_gw_warmup_start", 0)),
        steps=int(getattr(params, "lambda_gw_warmup_steps", 0)),
        target=float(getattr(params, "lambda_gw", 0.0)),
    )


def _metric_warmup_fraction(params: AttributeDict) -> float:
    return ours_train._linear_ramp(
        batch_idx=int(getattr(params, "batch_idx_train", 0)),
        start=int(getattr(params, "metric_warmup_start", 0)),
        steps=int(getattr(params, "metric_warmup_steps", 0)),
        target=1.0,
    )


def _effective_metric_grad_scale(params: AttributeDict) -> float:
    return float(getattr(params, "metric_grad_scale", 0.0)) * (
        _metric_warmup_fraction(params)
    )


def _ctc_loss_from_gated(
    params: AttributeDict,
    gated_log_probs: Tensor,
    supervision_segments: Tensor,
    decoding_graph: k2.Fsa,
) -> Tensor:
    dense_fsa_vec = k2.DenseFsaVec(
        gated_log_probs,
        supervision_segments,
        allow_truncate=params.subsampling_factor - 1,
    )
    return k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    )


def _encoder_aligned_acoustic_features(
    feature: Tensor,
    feature_lens: Tensor,
    orig_idx: int,
    encoder_len: int,
    subsampling_factor: int,
) -> Tensor:
    raw_len = int(feature_lens[orig_idx].item())
    raw_sub_idx = (
        torch.arange(encoder_len, device=feature.device) * subsampling_factor
        + subsampling_factor // 2
    )
    raw_sub_idx = raw_sub_idx.clamp(max=max(raw_len - 1, 0))
    return feature[orig_idx, raw_sub_idx, :]


def _make_bpe_length_batch(
    token_ids: List[List[int]],
    graph_compiler: BpeCtcTrainingGraphCompiler,
    device: torch.device,
    U_max: int,
) -> Tensor:
    bpe_rows = []
    for ids in token_ids:
        row = ours_train._compute_bpe_lengths(ids, graph_compiler.sp, device)
        if row.numel() < U_max:
            row = torch.cat([row, row.new_ones(U_max - row.numel())], dim=0)
        bpe_rows.append(row)
    return torch.stack(bpe_rows, dim=0)


def _compute_fgw_alignment_loss(
    params: AttributeDict,
    lp_sorted: Tensor,
    alpha_eff_sorted: Tensor,
    labels_sorted: Tensor,
    flen_sorted: Tensor,
    llen_sorted: Tensor,
    token_ids: List[List[int]],
    graph_compiler: GraphCompiler,
    feature: Tensor,
    feature_lens: Tensor,
    sorted_to_orig: List[int],
    supervisions: Dict[str, Any],
    structural_metric: Optional[nn.Module],
    metric_target_occupancy: Optional[Tensor],
    is_training: bool,
    debug: bool,
) -> Tuple[
    Tensor,
    Optional[Tensor],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Tensor]],
]:
    B, T_max, _ = lp_sorted.shape
    U_max = labels_sorted.size(1)
    device = lp_sorted.device
    align_loss_type = getattr(params, "align_loss_type", "plan-cost")
    need_full_plan = (
        align_loss_type == "fb-ce"
        or float(getattr(params, "lambda_plan_w1", 0.0)) > 0.0
        or metric_target_occupancy is not None
    )
    full_plan = (
        lp_sorted.new_zeros((B, T_max, U_max)) if need_full_plan else None
    )
    loss_rows = []
    metric_diagnostic_rows: List[Dict[str, Tensor]] = []
    debug_info: Optional[Dict[str, Any]] = None

    if params.col_marginal_type == "bpe":
        if not isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
            raise ValueError(
                "--col-marginal-type=bpe is not valid for phone targets; "
                "use acoustic or uniform."
            )
        bpe_lengths_b = _make_bpe_length_batch(
            token_ids=token_ids,
            graph_compiler=graph_compiler,
            device=device,
            U_max=U_max,
        )
    else:
        bpe_lengths_b = None

    for sorted_idx, ids in enumerate(token_ids):
        L = int(flen_sorted[sorted_idx].item())
        U = int(llen_sorted[sorted_idx].item())
        if L <= 0 or U <= 0:
            loss_rows.append(lp_sorted.new_tensor(0.0))
            continue

        orig_idx = int(sorted_to_orig[sorted_idx])
        acoustic_features = _encoder_aligned_acoustic_features(
            feature=feature,
            feature_lens=feature_lens,
            orig_idx=orig_idx,
            encoder_len=L,
            subsampling_factor=params.subsampling_factor,
        )
        labels_i = labels_sorted[sorted_idx, :U]
        bpe_lengths_i = (
            bpe_lengths_b[sorted_idx, :U] if bpe_lengths_b is not None else None
        )
        need_plan = need_full_plan or (debug and sorted_idx == 0)
        out = vi_fgw_loss_v2(
            log_p_nonblank=lp_sorted[sorted_idx, :L],
            alpha=alpha_eff_sorted[sorted_idx, :L],
            labels=labels_i,
            acoustic_features=acoustic_features,
            bpe_lengths=bpe_lengths_i,
            column_marginal_type=params.col_marginal_type,
            alpha_smooth_mix=params.alpha_smooth_mix,
            bpe_col_floor=params.bpe_col_floor,
            token_prior_sigma=params.ot_token_prior_sigma,
            token_prior_score_temp=params.ot_token_prior_score_temp,
            token_prior_floor=params.ot_token_prior_floor,
            eps=params.ot_eps,
            iters=params.ot_iters,
            beta_pos=params.ot_beta_pos,
            lambda_gw=_effective_lambda_gw(params),
            n_outer=params.gw_n_outer,
            frame_metric=structural_metric,
            metric_rho=float(params.metric_rho),
            metric_normalization=str(params.metric_normalization),
            metric_grad_scale=(
                _effective_metric_grad_scale(params) if is_training else 0.0
            ),
            metric_surrogate=str(params.metric_surrogate),
            metric_target_occupancy=(
                metric_target_occupancy[sorted_idx, :L, :U]
                if metric_target_occupancy is not None
                else None
            ),
            metric_occ_w1_weight=float(params.metric_occ_w1_weight),
            metric_occ_barycenter_weight=float(
                params.metric_occ_barycenter_weight
            ),
            metric_occ_log_std_weight=float(params.metric_occ_log_std_weight),
            metric_moment_reg_weight=(
                float(params.metric_moment_reg_weight)
                * _metric_warmup_fraction(params)
                if is_training
                else 0.0
            ),
            metric_identity_reg_weight=(
                float(params.metric_identity_reg_weight)
                * _metric_warmup_fraction(params)
                if is_training
                else 0.0
            ),
            return_plan=need_plan,
        )

        if need_plan:
            loss_i, plan_i = out
            if plan_i is not None and full_plan is not None:
                full_plan[sorted_idx, :L, :U] = plan_i
            if debug and sorted_idx == 0 and plan_i is not None:
                if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
                    token_pieces = [graph_compiler.sp.id_to_piece(t) for t in ids]
                else:
                    token_pieces = [
                        graph_compiler.token_table.get(t) for t in ids
                    ]
                cuts = supervisions.get("cut", None)
                cut_id = None
                if cuts is not None and orig_idx < len(cuts):
                    cut_id = getattr(cuts[orig_idx], "id", None)
                debug_info = {
                    "P": plan_i,
                    "token_pieces": token_pieces,
                    "cut_id": cut_id,
                }
            if plan_i is not None and metric_target_occupancy is not None:
                _, metric_row = metric_occupancy_geometry_loss(
                    plan=plan_i,
                    target_occupancy=metric_target_occupancy[
                        sorted_idx, :L, :U
                    ],
                    w1_weight=float(params.metric_occ_w1_weight),
                    barycenter_weight=float(
                        params.metric_occ_barycenter_weight
                    ),
                    log_std_weight=float(params.metric_occ_log_std_weight),
                )
                metric_diagnostic_rows.append(metric_row)
        else:
            loss_i = out
        loss_rows.append(loss_i)

    plan_cost_loss = torch.stack(loss_rows, dim=0)
    metric_diagnostics = None
    if metric_diagnostic_rows:
        metric_diagnostics = {
            key: torch.stack([row[key] for row in metric_diagnostic_rows]).mean()
            for key in metric_diagnostic_rows[0]
        }
    return plan_cost_loss, full_plan, debug_info, metric_diagnostics


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    graph_compiler: GraphCompiler,
    is_training: bool,
    warmup: float = 1.0,
    debug: bool = False,
) -> Tuple[Tensor, MetricsTracker, Optional[Dict[str, Any]]]:
    """VFTA loss with the OT plan replaced by an FGW plan when enabled."""
    if float(getattr(params, "lambda_gw", 0.0)) <= 0.0:
        loss, info, debug_info = _BASE_COMPUTE_LOSS(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=is_training,
            warmup=warmup,
            debug=debug,
        )
        if "frames" in info:
            info["lambda_gw"] = 0.0
            info["gw_n_outer"] = 0.0
        return loss, info, debug_info

    if not isinstance(
        graph_compiler,
        (BpeCtcTrainingGraphCompiler, WordPhoneCtcTrainingGraphCompiler),
    ):
        raise ValueError("VFTA+FGW requires BPE or deterministic phone targets.")

    model_ref = model.module if hasattr(model, "module") else model
    device = next(model_ref.parameters()).device
    feature = batch["inputs"].to(device)
    assert feature.ndim == 3

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )
    token_ids = graph_compiler.texts_to_ids(texts)
    decoding_graph = graph_compiler.compile(token_ids)

    targets_padded, target_lengths_t = ours_train._build_padded_targets(
        token_ids, device
    )
    sorted_to_orig = [int(i) for i in supervision_segments[:, 0].tolist()]
    orig_order = torch.zeros(len(token_ids), dtype=torch.long, device=device)
    for sorted_idx, orig_idx in enumerate(sorted_to_orig):
        orig_order[orig_idx] = sorted_idx
    targets_orig = targets_padded[orig_order]
    tlen_orig = target_lengths_t[orig_order]

    mix = ours_train._gate_mix(
        batch_idx=params.batch_idx_train,
        warmup_start=params.gate_warmup_start,
        warmup_steps=params.gate_warmup_steps,
    )
    gate_ctc_mode = getattr(params, "gate_ctc_mode", "mixed")
    dual_ctc_posterior_target = float(
        getattr(params, "dual_ctc_posterior_weight", 0.5)
    )
    effective_mix = mix if is_training or gate_ctc_mode == "mixed" else 0.0

    with torch.set_grad_enabled(is_training):
        (
            encoder_memory,
            memory_mask,
            encoder_out_lens,
            log_p_nonblank,
            alpha_prior_batch,
            alpha_post_batch,
        ) = model(
            feature,
            supervisions,
            targets=targets_orig,
            target_lengths=tlen_orig,
            warmup=warmup,
        )

        train_prior_logit_bias = ours_train._linear_ramp(
            batch_idx=params.batch_idx_train,
            start=params.train_prior_bias_start,
            steps=params.train_prior_bias_steps,
            target=float(params.train_prior_logit_bias),
        )
        if train_prior_logit_bias != 0.0:
            alpha_prior_batch = alpha_prior_batch.float().clamp(
                1.0e-5, 1.0 - 1.0e-5
            )
            alpha_prior_batch = torch.sigmoid(
                torch.logit(alpha_prior_batch) + train_prior_logit_bias
            ).to(log_p_nonblank.dtype)

        train_prior_mix = min(max(float(params.train_prior_mix), 0.0), 1.0)
        if gate_ctc_mode == "dual":
            alpha_eff_batch = (
                (1.0 - effective_mix) * alpha_prior_batch
                + effective_mix * alpha_post_batch
            )
            ctc_post_weight = effective_mix * dual_ctc_posterior_target
            ctc_prior_weight = 1.0 - ctc_post_weight
        else:
            post_weight = mix * (1.0 - train_prior_mix)
            prior_weight = 1.0 - post_weight
            alpha_eff_batch = (
                prior_weight * alpha_prior_batch
                + post_weight * alpha_post_batch
            )
            ctc_post_weight = 0.0
            ctc_prior_weight = 1.0

        nnet_output_gated = build_gated_log_probs_v2(
            log_p_nonblank,
            alpha_eff_batch,
        )
        if gate_ctc_mode == "dual":
            nnet_output_prior = build_gated_log_probs_v2(
                log_p_nonblank,
                alpha_prior_batch,
            )
            nnet_output_post = build_gated_log_probs_v2(
                log_p_nonblank,
                alpha_post_batch,
            )
        else:
            nnet_output_prior = nnet_output_gated
            nnet_output_post = nnet_output_gated

    T_out = log_p_nonblank.size(1)
    frame_idx = torch.arange(T_out, device=device).unsqueeze(0)
    valid_mask = frame_idx < encoder_out_lens.unsqueeze(1)

    ctc_loss_prior = _ctc_loss_from_gated(
        params=params,
        gated_log_probs=nnet_output_prior,
        supervision_segments=supervision_segments,
        decoding_graph=decoding_graph,
    )
    if gate_ctc_mode == "dual" and ctc_post_weight > 0.0:
        ctc_loss_post = _ctc_loss_from_gated(
            params=params,
            gated_log_probs=nnet_output_post,
            supervision_segments=supervision_segments,
            decoding_graph=decoding_graph,
        )
        ctc_loss = (
            ctc_prior_weight * ctc_loss_prior
            + ctc_post_weight * ctc_loss_post
        )
    else:
        ctc_loss_post = torch.zeros_like(ctc_loss_prior)
        posterior_zero_dependency = (
            0.0 * alpha_post_batch.sum() if is_training else 0.0
        )
        ctc_loss = ctc_loss_prior + posterior_zero_dependency
    ctc_loss_is_finite = torch.isfinite(ctc_loss)
    ctc_loss_is_finite = ctc_loss_is_finite & torch.isfinite(ctc_loss_prior)
    if ctc_post_weight > 0.0:
        ctc_loss_is_finite = ctc_loss_is_finite & torch.isfinite(ctc_loss_post)

    lambda_ot = params.lambda_ot
    lambda_kl = params.lambda_kl_blank
    lambda_alpha_mean = params.lambda_alpha_mean
    lambda_plan_w1 = (
        ours_train._linear_ramp(
            batch_idx=params.batch_idx_train,
            start=params.plan_w1_warmup_start,
            steps=params.plan_w1_warmup_steps,
            target=float(params.lambda_plan_w1),
        )
        if is_training
        else 0.0
    )
    debug_info: Optional[Dict[str, Any]] = None

    sorted_index = torch.tensor(sorted_to_orig, device=device, dtype=torch.long)
    lp_sorted = log_p_nonblank[sorted_index]
    alpha_eff_sorted = alpha_eff_batch[sorted_index]
    alpha_post_sorted = alpha_post_batch[sorted_index]
    alpha_prior_sorted = alpha_prior_batch[sorted_index]
    flen_sorted = encoder_out_lens[sorted_index]
    labels_sorted = targets_padded
    llen_sorted = target_lengths_t

    structural_metric = getattr(model_ref, "structural_metric", None)
    metric_uses_ctc_occupancy = (
        structural_metric is not None
        and str(getattr(params, "metric_surrogate", "token-cost"))
        == "ctc-occupancy"
        and is_training
        and _effective_metric_grad_scale(params) > 0.0
    )

    # Reuse the same differentiable prior/effective CTC occupancy for the
    # reciprocal plan-to-CTC loss when it is active.  If only the metric-side
    # target needs it, construct it without autograd: the target is fixed by
    # design and must not leak gradients into the classifier or encoder.
    precomputed_prior_occupancy: Optional[Tensor] = None
    precomputed_effective_occupancy: Optional[Tensor] = None
    need_plan_w1_occupancy = lambda_plan_w1 > 0.0
    need_prior_occupancy = metric_uses_ctc_occupancy or (
        need_plan_w1_occupancy and gate_ctc_mode == "dual"
    )
    need_effective_occupancy = metric_uses_ctc_occupancy or (
        need_plan_w1_occupancy and gate_ctc_mode != "dual"
    )
    occupancy_requires_grad = is_training and need_plan_w1_occupancy
    if need_prior_occupancy and gate_ctc_mode == "dual":
        with torch.set_grad_enabled(occupancy_requires_grad):
            precomputed_prior_occupancy = ctc_token_occupancy_batched(
                log_probs=nnet_output_prior[sorted_index],
                labels=labels_sorted,
                frame_lens=flen_sorted,
                label_lens=llen_sorted,
                blank_id=0,
            )
    if need_effective_occupancy and gate_ctc_mode != "dual":
        with torch.set_grad_enabled(occupancy_requires_grad):
            precomputed_effective_occupancy = ctc_token_occupancy_batched(
                log_probs=nnet_output_gated[sorted_index],
                labels=labels_sorted,
                frame_lens=flen_sorted,
                label_lens=llen_sorted,
                blank_id=0,
            )
    metric_target_occupancy = (
        precomputed_prior_occupancy
        if gate_ctc_mode == "dual"
        else precomputed_effective_occupancy
    )

    T_max = lp_sorted.size(1)
    valid_mask_sorted = (
        torch.arange(T_max, device=device)[None, :] < flen_sorted[:, None]
    )

    if is_training and mix > 0.0:
        kl_eps = 1.0e-5
        p = alpha_post_sorted.detach().float().clamp(kl_eps, 1.0 - kl_eps)
        q = alpha_prior_sorted.float().clamp(kl_eps, 1.0 - kl_eps)
        kl_per_frame = p * (p.log() - q.log()) + (1.0 - p) * (
            (1.0 - p).log() - (1.0 - q).log()
        )
        m = valid_mask_sorted.float()
        kl_loss = (kl_per_frame * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    else:
        kl_loss = lp_sorted.new_zeros(lp_sorted.size(0))

    if lambda_ot > 0 or lambda_plan_w1 > 0:
        # The model forward above is already guarded by ``is_training``, but the
        # learnable frame metric is evaluated inside the FGW loss.  Guard this
        # path too so validation does not build a graph solely through L and
        # violate the validation contract that the returned loss is detached.
        with torch.set_grad_enabled(is_training):
            (
                plan_cost_loss,
                plan,
                debug_info,
                metric_occupancy_diagnostics,
            ) = _compute_fgw_alignment_loss(
                params=params,
                lp_sorted=lp_sorted,
                alpha_eff_sorted=alpha_eff_sorted,
                labels_sorted=labels_sorted,
                flen_sorted=flen_sorted,
                llen_sorted=llen_sorted,
                token_ids=token_ids,
                graph_compiler=graph_compiler,
                feature=feature,
                feature_lens=feature_lens,
                sorted_to_orig=sorted_to_orig,
                supervisions=supervisions,
                structural_metric=structural_metric,
                metric_target_occupancy=metric_target_occupancy,
                is_training=is_training,
                debug=debug,
            )
        align_loss_type = getattr(params, "align_loss_type", "plan-cost")
        if align_loss_type == "fb-ce":
            assert plan is not None
            ot_loss = fb_posterior_consistency_loss(
                plan=plan,
                gated_log_probs=nnet_output_gated[sorted_index],
                targets=labels_sorted,
                input_lengths=flen_sorted,
                target_lengths=llen_sorted,
            )
        else:
            ot_loss = plan_cost_loss

        if lambda_plan_w1 > 0:
            assert plan is not None
            plan_w1_loss, plan_w1_prior_loss, plan_w1_post_loss = (
                plan_w1_loss_for_gates(
                    teacher_plan=plan,
                    effective_log_probs=nnet_output_gated[sorted_index],
                    prior_log_probs=nnet_output_prior[sorted_index],
                    posterior_log_probs=nnet_output_post[sorted_index],
                    labels=labels_sorted,
                    frame_lens=flen_sorted,
                    label_lens=llen_sorted,
                    dual_gate=gate_ctc_mode == "dual",
                    prior_weight=ctc_prior_weight,
                    posterior_weight=ctc_post_weight,
                    blank_id=0,
                    precomputed_effective_occupancy=(
                        precomputed_effective_occupancy
                    ),
                    precomputed_prior_occupancy=precomputed_prior_occupancy,
                )
            )
        else:
            plan_w1_loss = lp_sorted.new_zeros(lp_sorted.size(0))
            plan_w1_prior_loss = lp_sorted.new_zeros(lp_sorted.size(0))
            plan_w1_post_loss = lp_sorted.new_zeros(lp_sorted.size(0))
    else:
        metric_occupancy_diagnostics = None
        ot_loss = lp_sorted.new_zeros(lp_sorted.size(0))
        plan_w1_loss = lp_sorted.new_zeros(lp_sorted.size(0))
        plan_w1_prior_loss = lp_sorted.new_zeros(lp_sorted.size(0))
        plan_w1_post_loss = lp_sorted.new_zeros(lp_sorted.size(0))

    alpha_mean_source = getattr(params, "alpha_mean_source", "eff")
    if alpha_mean_source == "prior":
        alpha_mean_for_loss = alpha_prior_batch.float()
    else:
        alpha_mean_for_loss = alpha_eff_batch.float()
    alpha_mean = ours_train._valid_frame_mean(alpha_mean_for_loss, valid_mask)
    alpha_mean_target = alpha_mean.new_tensor(float(params.alpha_mean_target))
    alpha_mean_mode = getattr(params, "alpha_mean_mode", "ceiling")
    if alpha_mean_mode == "floor":
        alpha_mean_loss = torch.relu(alpha_mean_target - alpha_mean)
    elif alpha_mean_mode == "ceiling":
        alpha_mean_loss = torch.relu(alpha_mean - alpha_mean_target)
    else:
        raise ValueError(f"Unsupported alpha_mean_mode: {alpha_mean_mode}")
    alpha_mean_loss_total = alpha_mean_loss * valid_mask.sum().to(alpha_mean_loss.dtype)

    if not torch.all(ctc_loss_is_finite):
        ctc_loss = ctc_loss[ctc_loss_is_finite]
        ctc_loss_prior = ctc_loss_prior[ctc_loss_is_finite]
        ctc_loss_post = ctc_loss_post[ctc_loss_is_finite]
        kl_loss = kl_loss[ctc_loss_is_finite]
        ot_loss = ot_loss[ctc_loss_is_finite]
        plan_w1_loss = plan_w1_loss[ctc_loss_is_finite]
        plan_w1_prior_loss = plan_w1_prior_loss[ctc_loss_is_finite]
        plan_w1_post_loss = plan_w1_post_loss[ctc_loss_is_finite]
        if torch.all(~ctc_loss_is_finite):
            raise ValueError("All losses are inf/nan; reduce max-duration.")

    metric_spectrum_objective = lp_sorted.new_tensor(0.0)
    metric_spectrum_diagnostics: Optional[Dict[str, Tensor]] = None
    if structural_metric is not None:
        spectrum_weight = float(params.metric_spectrum_reg_weight)
        spectrum_warmup = _metric_warmup_fraction(params) if is_training else 0.0
        if is_training and spectrum_weight > 0.0 and spectrum_warmup > 0.0:
            spectrum_reg, metric_spectrum_diagnostics = (
                structural_metric.spectrum_regularizer(
                    max_eigenvalue=float(
                        params.metric_spectrum_max_eigenvalue
                    )
                )
            )
            # Other per-utterance structural losses are summed, so preserve
            # the regularizer's relative weight across different batch sizes.
            metric_spectrum_objective = (
                spectrum_weight
                * spectrum_warmup
                * spectrum_reg
                * max(int(ctc_loss.numel()), 1)
            )
        else:
            with torch.no_grad():
                _, metric_spectrum_diagnostics = (
                    structural_metric.spectrum_regularizer(
                        max_eigenvalue=float(
                            params.metric_spectrum_max_eigenvalue
                        )
                    )
                )

    if params.att_rate != 0.0:
        with torch.set_grad_enabled(is_training):
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = model_ref.encoder.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        loss = (
            (1.0 - params.att_rate) * ctc_loss.sum()
            + params.att_rate * att_loss
            + lambda_kl * kl_loss.sum()
            + lambda_ot * ot_loss.sum()
            + lambda_ot * metric_spectrum_objective
            + lambda_plan_w1 * plan_w1_loss.sum()
            + lambda_alpha_mean * alpha_mean_loss_total
        )
    else:
        loss = (
            (1.0 - params.att_rate) * ctc_loss.sum()
            + lambda_kl * kl_loss.sum()
            + lambda_ot * ot_loss.sum()
            + lambda_ot * metric_spectrum_objective
            + lambda_plan_w1 * plan_w1_loss.sum()
            + lambda_alpha_mean * alpha_mean_loss_total
        )
        att_loss = log_p_nonblank.new_tensor(0.0)

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = encoder_out_lens.sum().detach().cpu().item()
    valid_frames = valid_mask.sum().clamp_min(1).float()
    metric_frames = info["frames"]
    info["ctc_loss"] = ctc_loss.sum().detach().cpu().item()
    info["ctc_loss_prior"] = ctc_loss_prior.sum().detach().cpu().item()
    info["ctc_loss_post"] = ctc_loss_post.sum().detach().cpu().item()
    info["ctc_prior_weight"] = ctc_prior_weight * metric_frames
    info["ctc_post_weight"] = ctc_post_weight * metric_frames
    info["kl_blank_loss"] = kl_loss.sum().detach().cpu().item()
    info["ot_loss"] = ot_loss.sum().detach().cpu().item()
    info["plan_w1_loss"] = plan_w1_loss.sum().detach().cpu().item()
    info["plan_w1_prior_loss"] = (
        plan_w1_prior_loss.sum().detach().cpu().item()
    )
    info["plan_w1_post_loss"] = (
        plan_w1_post_loss.sum().detach().cpu().item()
    )
    info["lambda_plan_w1"] = lambda_plan_w1 * metric_frames
    info["lambda_gw"] = _effective_lambda_gw(params) * metric_frames
    info["lambda_gw_target"] = float(params.lambda_gw) * metric_frames
    info["gw_n_outer"] = float(params.gw_n_outer) * metric_frames
    if structural_metric is not None:
        with torch.no_grad():
            weight = structural_metric.effective_projection_weight(
                rho=float(params.metric_rho)
            ).float()
            identity = torch.eye(
                weight.size(0), dtype=weight.dtype, device=weight.device
            )
            metric_delta = (weight - identity).square().mean().sqrt()
            gram = weight.transpose(0, 1) @ weight
            metric_identity_error = (gram - identity).square().mean().sqrt()
        info["metric_delta_fro"] = metric_delta.cpu().item() * metric_frames
        info["metric_identity_error"] = (
            metric_identity_error.cpu().item() * metric_frames
        )
        info["metric_rho"] = float(params.metric_rho) * metric_frames
        info["metric_grad_scale"] = (
            _effective_metric_grad_scale(params) * metric_frames
        )
        info["metric_warmup_fraction"] = (
            _metric_warmup_fraction(params) * metric_frames
        )
        if metric_spectrum_diagnostics is not None:
            for key, value in metric_spectrum_diagnostics.items():
                info[key] = value.detach().cpu().item() * metric_frames
        info["metric_spectrum_reg_weight"] = (
            float(params.metric_spectrum_reg_weight) * metric_frames
        )
        if metric_occupancy_diagnostics is not None:
            for key, value in metric_occupancy_diagnostics.items():
                info[key] = value.detach().cpu().item() * metric_frames
        else:
            for key in (
                "metric_occ_w1",
                "metric_occ_barycenter",
                "metric_occ_log_std",
                "metric_occ_loss",
            ):
                info[key] = 0.0
    info["alpha_mean_loss"] = alpha_mean_loss.detach().cpu().item() * metric_frames
    info["gate_mix"] = effective_mix * metric_frames
    info["train_prior_mix"] = (
        min(max(float(params.train_prior_mix), 0.0), 1.0) * metric_frames
    )
    info["train_prior_logit_bias"] = train_prior_logit_bias * metric_frames
    info["alpha_prior_mean"] = (
        (alpha_prior_batch * valid_mask.to(alpha_prior_batch.dtype)).sum()
        / valid_frames
    ).detach().cpu().item() * metric_frames
    info["alpha_post_mean"] = (
        (alpha_post_batch * valid_mask.to(alpha_post_batch.dtype)).sum()
        / valid_frames
    ).detach().cpu().item() * metric_frames
    info["alpha_eff_mean"] = (
        (alpha_eff_batch * valid_mask.to(alpha_eff_batch.dtype)).sum()
        / valid_frames
    ).detach().cpu().item() * metric_frames
    info["alpha_prior_frame_std"] = (
        mean_valid_frame_std(alpha_prior_batch, encoder_out_lens)
        .detach()
        .cpu()
        .item()
        * metric_frames
    )
    info["alpha_post_frame_std"] = (
        mean_valid_frame_std(alpha_post_batch, encoder_out_lens)
        .detach()
        .cpu()
        .item()
        * metric_frames
    )
    info["alpha_eff_frame_std"] = (
        mean_valid_frame_std(alpha_eff_batch, encoder_out_lens)
        .detach()
        .cpu()
        .item()
        * metric_frames
    )
    info["alpha_prior_post_absdiff"] = (
        mean_valid_abs_diff(alpha_post_batch, alpha_prior_batch, encoder_out_lens)
        .detach()
        .cpu()
        .item()
        * metric_frames
    )
    if params.att_rate != 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()
    info["loss"] = loss.detach().cpu().item()
    info["utterances"] = feature.size(0)
    info["utt_duration"] = feature_lens.sum().item()
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    return loss, info, debug_info


def run(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if not 0.0 <= args.lambda_gw <= 1.0:
        raise ValueError("--lambda-gw must be between 0 and 1")
    if args.gw_n_outer < 0:
        raise ValueError("--gw-n-outer must be non-negative")
    if not 0.0 <= args.metric_rho <= 1.0:
        raise ValueError("--metric-rho must be between 0 and 1")
    if args.frame_metric != "fixed-cosine" and args.gw_n_outer < 1:
        raise ValueError("a learned frame metric requires --gw-n-outer >= 1")
    if args.metric_max_log_scale <= 0.0:
        raise ValueError("--metric-max-log-scale must be positive")
    if args.metric_grad_scale < 0.0:
        raise ValueError("--metric-grad-scale must be non-negative")
    if min(
        args.metric_occ_w1_weight,
        args.metric_occ_barycenter_weight,
        args.metric_occ_log_std_weight,
    ) < 0.0:
        raise ValueError("metric occupancy weights must be non-negative")
    if args.metric_lr_scale <= 0.0:
        raise ValueError("--metric-lr-scale must be positive")
    if args.metric_warmup_start < 0 or args.metric_warmup_steps < 0:
        raise ValueError("metric warmup start/steps must be non-negative")
    if args.metric_moment_reg_weight < 0.0:
        raise ValueError("--metric-moment-reg-weight must be non-negative")
    if args.metric_identity_reg_weight < 0.0:
        raise ValueError("--metric-identity-reg-weight must be non-negative")
    if args.metric_spectrum_reg_weight < 0.0:
        raise ValueError("--metric-spectrum-reg-weight must be non-negative")
    if args.metric_spectrum_max_eigenvalue <= 1.0:
        raise ValueError("--metric-spectrum-max-eigenvalue must be greater than 1")
    if args.keep_last_epoch_checkpoints < 0:
        raise ValueError("--keep-last-epoch-checkpoints must be non-negative")
    _parse_retained_epochs(args.retain_epoch_checkpoints)
    ours_train.compute_loss = compute_loss
    ours_train.configure_model = _configure_vfta_model
    ours_train.save_checkpoint = save_checkpoint
    ours_train.run(rank=rank, world_size=world_size, args=args)


def main() -> None:
    parser = get_parser()
    ours_train.LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    if args.world_size > 1:
        mp.spawn(run, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
