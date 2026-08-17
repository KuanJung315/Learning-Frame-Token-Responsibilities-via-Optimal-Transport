#!/usr/bin/env python3
"""Train conformer_ctc2 VFTA/VI with CR-CTC consistency regularization.

This is a compatibility recipe: the model, optimizer, checkpointing, data
module, and decode path are inherited from train_vi_ot_v2.py.  Only the
training objective is swapped so the existing VFTA/VI loss can be combined
with a CR-CTC style two-view consistency term.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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
from cr_ctc_utils import (  # noqa: E402
    cr_ctc_consistency_loss,
    linear_warmup_weight,
    make_cr_view,
    with_gradient_scale,
)
from ctc_occupancy import fb_posterior_consistency_loss  # noqa: E402
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler  # noqa: E402
from icefall.graph_compiler import CtcTrainingGraphCompiler  # noqa: E402
from icefall.utils import AttributeDict, MetricsTracker, encode_supervisions  # noqa: E402
from ot_prior_v2 import vi_ot_loss_v2, vi_ot_loss_v2_batched  # noqa: E402
from varctc_v2_utils import (  # noqa: E402
    build_gated_log_probs_v2,
    mean_valid_abs_diff,
    mean_valid_frame_std,
)


GraphCompiler = Union[BpeCtcTrainingGraphCompiler, CtcTrainingGraphCompiler]


def get_parser() -> argparse.ArgumentParser:
    parser = ours_train.get_parser()
    parser.set_defaults(exp_dir="conformer_ctc2/cr_ctc_ours/exp_li100h")

    parser.add_argument(
        "--cr-ctc-weight",
        type=float,
        default=0.1,
        help="Weight for the CR-CTC frame-distribution consistency loss.",
    )
    parser.add_argument(
        "--cr-ctc-supervised-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the second augmented view's supervised CTC loss. "
            "The two CTC losses are averaged by 1 + this value, so the "
            "overall CTC scale stays close to the original recipe."
        ),
    )
    parser.add_argument(
        "--cr-stop-gradient",
        type=ours_train.str2bool,
        default=True,
        help=(
            "Use symmetric stop-gradient KL for the consistency term. "
            "This matches the self-distillation view of CR-CTC."
        ),
    )
    parser.add_argument(
        "--cr-temperature",
        type=float,
        default=1.0,
        help="Temperature applied before the CR-CTC KL computation.",
    )
    parser.add_argument(
        "--cr-gate-grad-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the consistency gradient entering the VFTA blank gate. "
            "The forward distributions are unchanged."
        ),
    )
    parser.add_argument(
        "--cr-warmup-start",
        type=int,
        default=0,
        help="Batch at which the CR consistency-weight ramp starts.",
    )
    parser.add_argument(
        "--cr-warmup-steps",
        type=int,
        default=0,
        help="Batches used to linearly ramp CR weight to --cr-ctc-weight.",
    )
    parser.add_argument(
        "--cr-num-time-masks",
        type=int,
        default=2,
        help="Number of extra time masks used to create the second view.",
    )
    parser.add_argument(
        "--cr-time-mask-size",
        type=int,
        default=80,
        help="Maximum width of each extra time mask.",
    )
    parser.add_argument(
        "--cr-num-feature-masks",
        type=int,
        default=2,
        help="Number of extra feature masks used to create the second view.",
    )
    parser.add_argument(
        "--cr-feature-mask-size",
        type=int,
        default=27,
        help="Maximum width of each extra feature mask.",
    )
    parser.add_argument(
        "--cr-mask-value",
        type=float,
        default=0.0,
        help="Feature value used inside the extra CR-CTC masks.",
    )
    return parser


def _build_gated_outputs(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    feature: Tensor,
    supervisions: dict,
    targets_orig: Tensor,
    tlen_orig: Tensor,
    warmup: float,
    batch_idx: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float]:
    mix = ours_train._gate_mix(
        batch_idx=batch_idx,
        warmup_start=params.gate_warmup_start,
        warmup_steps=params.gate_warmup_steps,
    )
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
        batch_idx=batch_idx,
        start=params.train_prior_bias_start,
        steps=params.train_prior_bias_steps,
        target=float(params.train_prior_logit_bias),
    )
    if train_prior_logit_bias != 0.0:
        alpha_prior_batch = alpha_prior_batch.float().clamp(1.0e-5, 1.0 - 1.0e-5)
        alpha_prior_batch = torch.sigmoid(
            torch.logit(alpha_prior_batch) + train_prior_logit_bias
        ).to(log_p_nonblank.dtype)

    train_prior_mix = min(max(float(params.train_prior_mix), 0.0), 1.0)
    post_weight = mix * (1.0 - train_prior_mix)
    prior_weight = 1.0 - post_weight
    alpha_eff_batch = prior_weight * alpha_prior_batch + post_weight * alpha_post_batch
    nnet_output_gated = build_gated_log_probs_v2(log_p_nonblank, alpha_eff_batch)

    return (
        encoder_memory,
        memory_mask,
        encoder_out_lens,
        log_p_nonblank,
        alpha_prior_batch,
        alpha_post_batch,
        alpha_eff_batch,
        nnet_output_gated,
        train_prior_logit_bias,
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


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    graph_compiler: GraphCompiler,
    is_training: bool,
    warmup: float = 1.0,
    debug: bool = False,
) -> Tuple[Tensor, MetricsTracker, Optional[Dict[str, Any]]]:
    """VFTA/VI loss plus CR-CTC consistency between two augmented views."""
    if not isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
        raise ValueError("CR-CTC+Ours currently requires a lang_bpe directory.")

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
    sorted_to_orig = supervision_segments[:, 0].tolist()
    orig_order = torch.zeros(len(token_ids), dtype=torch.long, device=device)
    for sorted_idx, orig_idx in enumerate(sorted_to_orig):
        orig_order[int(orig_idx)] = sorted_idx
    targets_orig = targets_padded[orig_order]
    tlen_orig = target_lengths_t[orig_order]

    mix = ours_train._gate_mix(
        batch_idx=params.batch_idx_train,
        warmup_start=params.gate_warmup_start,
        warmup_steps=params.gate_warmup_steps,
    )

    with torch.set_grad_enabled(is_training):
        (
            encoder_memory,
            memory_mask,
            encoder_out_lens,
            log_p_nonblank,
            alpha_prior_batch,
            alpha_post_batch,
            alpha_eff_batch,
            nnet_output_gated,
            train_prior_logit_bias,
        ) = _build_gated_outputs(
            params=params,
            model=model,
            feature=feature,
            supervisions=supervisions,
            targets_orig=targets_orig,
            tlen_orig=tlen_orig,
            warmup=warmup,
            batch_idx=params.batch_idx_train,
        )

        use_cr = is_training and params.cr_ctc_weight > 0.0
        if use_cr:
            feature_cr = make_cr_view(feature, feature_lens, params)
            (
                _,
                _,
                encoder_out_lens_cr,
                log_p_nonblank_cr,
                _,
                _,
                alpha_eff_batch_cr,
                nnet_output_gated_cr,
                _,
            ) = _build_gated_outputs(
                params=params,
                model=model,
                feature=feature_cr,
                supervisions=supervisions,
                targets_orig=targets_orig,
                tlen_orig=tlen_orig,
                warmup=warmup,
                batch_idx=params.batch_idx_train,
            )
        else:
            encoder_out_lens_cr = encoder_out_lens
            log_p_nonblank_cr = log_p_nonblank
            alpha_eff_batch_cr = alpha_eff_batch
            nnet_output_gated_cr = nnet_output_gated

    t_out = log_p_nonblank.size(1)
    frame_idx = torch.arange(t_out, device=device).unsqueeze(0)
    valid_mask = frame_idx < encoder_out_lens.unsqueeze(1)

    ctc_loss = _ctc_loss_from_gated(
        params=params,
        gated_log_probs=nnet_output_gated,
        supervision_segments=supervision_segments,
        decoding_graph=decoding_graph,
    )
    ctc_loss_is_finite = torch.isfinite(ctc_loss)

    cr_supervised_weight = (
        float(params.cr_ctc_supervised_weight) if is_training else 0.0
    )
    if use_cr and cr_supervised_weight > 0.0:
        ctc_loss_cr = _ctc_loss_from_gated(
            params=params,
            gated_log_probs=nnet_output_gated_cr,
            supervision_segments=supervision_segments,
            decoding_graph=decoding_graph,
        )
        ctc_loss_is_finite = ctc_loss_is_finite & torch.isfinite(ctc_loss_cr)
        ctc_loss_for_model = (
            ctc_loss + cr_supervised_weight * ctc_loss_cr
        ) / (1.0 + cr_supervised_weight)
    else:
        ctc_loss_cr = ctc_loss.new_zeros(ctc_loss.shape)
        ctc_loss_for_model = ctc_loss

    if use_cr:
        cr_gate_grad_scale = float(params.cr_gate_grad_scale)
        if cr_gate_grad_scale == 1.0:
            consistency_log_probs = nnet_output_gated
            consistency_log_probs_cr = nnet_output_gated_cr
        else:
            consistency_log_probs = build_gated_log_probs_v2(
                log_p_nonblank,
                with_gradient_scale(alpha_eff_batch, cr_gate_grad_scale),
            )
            consistency_log_probs_cr = build_gated_log_probs_v2(
                log_p_nonblank_cr,
                with_gradient_scale(alpha_eff_batch_cr, cr_gate_grad_scale),
            )
        cr_consistency_loss = cr_ctc_consistency_loss(
            log_probs_a=consistency_log_probs,
            log_probs_b=consistency_log_probs_cr,
            lengths_a=encoder_out_lens,
            lengths_b=encoder_out_lens_cr,
            stop_gradient=bool(params.cr_stop_gradient),
            temperature=float(params.cr_temperature),
        )
    else:
        cr_gate_grad_scale = float(params.cr_gate_grad_scale)
        cr_consistency_loss = ctc_loss.new_zeros(ctc_loss.shape)

    effective_cr_weight = (
        linear_warmup_weight(
            batch_idx=params.batch_idx_train,
            start=int(params.cr_warmup_start),
            steps=int(params.cr_warmup_steps),
            target=float(params.cr_ctc_weight),
        )
        if is_training
        else 0.0
    )

    lambda_ot = params.lambda_ot
    lambda_kl = params.lambda_kl_blank
    lambda_alpha_mean = params.lambda_alpha_mean
    debug_info: Optional[Dict[str, Any]] = None

    sorted_index = torch.tensor(sorted_to_orig, device=device, dtype=torch.long)
    lp_sorted = log_p_nonblank[sorted_index]
    alpha_eff_sorted = alpha_eff_batch[sorted_index]
    alpha_post_sorted = alpha_post_batch[sorted_index]
    alpha_prior_sorted = alpha_prior_batch[sorted_index]
    flen_sorted = encoder_out_lens[sorted_index]
    labels_sorted = targets_padded
    llen_sorted = target_lengths_t

    t_max = lp_sorted.size(1)
    valid_mask_sorted = (
        torch.arange(t_max, device=device)[None, :] < flen_sorted[:, None]
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

    if lambda_ot > 0:
        if params.col_marginal_type == "bpe":
            u_max = labels_sorted.size(1)
            bpe_rows = []
            for ids in token_ids:
                row = ours_train._compute_bpe_lengths(
                    ids, graph_compiler.sp, device
                )
                if row.numel() < u_max:
                    row = torch.cat([row, row.new_ones(u_max - row.numel())], dim=0)
                bpe_rows.append(row)
            bpe_lengths_b = torch.stack(bpe_rows, dim=0)
        else:
            bpe_lengths_b = None

        align_loss_type = getattr(params, "align_loss_type", "plan-cost")
        plan_cost_loss, plan = vi_ot_loss_v2_batched(
            log_p_nonblank=lp_sorted,
            alpha=alpha_eff_sorted,
            labels=labels_sorted,
            frame_lens=flen_sorted,
            label_lens=llen_sorted,
            bpe_lengths=bpe_lengths_b,
            column_marginal_type=params.col_marginal_type,
            alpha_smooth_mix=params.alpha_smooth_mix,
            bpe_col_floor=params.bpe_col_floor,
            token_prior_sigma=params.ot_token_prior_sigma,
            token_prior_score_temp=params.ot_token_prior_score_temp,
            token_prior_floor=params.ot_token_prior_floor,
            eps=params.ot_eps,
            iters=params.ot_iters,
            beta_pos=params.ot_beta_pos,
            return_plan=True,
        )
        if align_loss_type == "fb-ce":
            ot_loss = fb_posterior_consistency_loss(
                plan=plan,
                gated_log_probs=nnet_output_gated[sorted_index],
                targets=labels_sorted,
                input_lengths=flen_sorted,
                target_lengths=llen_sorted,
            )
        else:
            ot_loss = plan_cost_loss

        if debug:
            ids0 = token_ids[0]
            l0 = int(flen_sorted[0].item())
            _, plan0 = vi_ot_loss_v2(
                log_p_nonblank=lp_sorted[0, :l0],
                alpha=alpha_eff_sorted[0, :l0],
                labels=torch.tensor(ids0, device=device, dtype=torch.long),
                bpe_lengths=(
                    bpe_lengths_b[0, : len(ids0)]
                    if bpe_lengths_b is not None
                    else None
                ),
                column_marginal_type=params.col_marginal_type,
                alpha_smooth_mix=params.alpha_smooth_mix,
                bpe_col_floor=params.bpe_col_floor,
                token_prior_sigma=params.ot_token_prior_sigma,
                token_prior_score_temp=params.ot_token_prior_score_temp,
                token_prior_floor=params.ot_token_prior_floor,
                eps=params.ot_eps,
                iters=params.ot_iters,
                beta_pos=params.ot_beta_pos,
                return_plan=True,
            )
            if plan0 is not None:
                token_pieces = [graph_compiler.sp.id_to_piece(t) for t in ids0]
                cuts = supervisions.get("cut", None)
                cut_id = None
                orig0 = int(sorted_to_orig[0])
                if cuts is not None and orig0 < len(cuts):
                    cut_id = getattr(cuts[orig0], "id", None)
                debug_info = {"P": plan0, "token_pieces": token_pieces, "cut_id": cut_id}
    else:
        ot_loss = lp_sorted.new_zeros(lp_sorted.size(0))

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
        ctc_loss_cr = ctc_loss_cr[ctc_loss_is_finite]
        ctc_loss_for_model = ctc_loss_for_model[ctc_loss_is_finite]
        kl_loss = kl_loss[ctc_loss_is_finite]
        ot_loss = ot_loss[ctc_loss_is_finite]
        cr_consistency_loss = cr_consistency_loss[ctc_loss_is_finite]
        if torch.all(~ctc_loss_is_finite):
            raise ValueError("All losses are inf/nan; reduce max-duration.")

    ctc_term = (1.0 - params.att_rate) * ctc_loss_for_model.sum()
    cr_term = effective_cr_weight * cr_consistency_loss.sum()

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
            ctc_term
            + params.att_rate * att_loss
            + lambda_kl * kl_loss.sum()
            + lambda_ot * ot_loss.sum()
            + lambda_alpha_mean * alpha_mean_loss_total
            + cr_term
        )
    else:
        loss = (
            ctc_term
            + lambda_kl * kl_loss.sum()
            + lambda_ot * ot_loss.sum()
            + lambda_alpha_mean * alpha_mean_loss_total
            + cr_term
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
    info["ctc_loss_cr"] = ctc_loss_cr.sum().detach().cpu().item()
    info["kl_blank_loss"] = kl_loss.sum().detach().cpu().item()
    info["ot_loss"] = ot_loss.sum().detach().cpu().item()
    info["cr_ctc_loss"] = cr_consistency_loss.sum().detach().cpu().item()
    info["cr_ctc_weight"] = float(params.cr_ctc_weight) * metric_frames
    info["cr_ctc_effective_weight"] = effective_cr_weight * metric_frames
    info["cr_gate_grad_scale"] = cr_gate_grad_scale * metric_frames
    info["cr_supervised_weight"] = cr_supervised_weight * metric_frames
    info["alpha_mean_loss"] = alpha_mean_loss.detach().cpu().item() * metric_frames
    info["gate_mix"] = mix * metric_frames
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
    if args.cr_ctc_weight < 0:
        raise ValueError("--cr-ctc-weight must be non-negative")
    if args.cr_ctc_supervised_weight < 0:
        raise ValueError("--cr-ctc-supervised-weight must be non-negative")
    if args.cr_temperature <= 0:
        raise ValueError("--cr-temperature must be positive")
    if not 0.0 <= args.cr_gate_grad_scale <= 1.0:
        raise ValueError("--cr-gate-grad-scale must be between 0 and 1")
    if args.cr_warmup_start < 0:
        raise ValueError("--cr-warmup-start must be non-negative")
    if args.cr_warmup_steps < 0:
        raise ValueError("--cr-warmup-steps must be non-negative")

    ours_train.compute_loss = compute_loss
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
