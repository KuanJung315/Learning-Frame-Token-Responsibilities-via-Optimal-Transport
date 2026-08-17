#!/usr/bin/env python3
"""Train the TIMIT Conformer with label-prior CTC.

This is the TIMIT counterpart of the local LibriSpeech reproduction of
Huang et al., "Less Peaky and More Accurate CTC Forced Alignment by Label
Priors".  CTC path scores use ``log y_k^t - alpha log P(k)`` and the unigram
prior is estimated from the preceding epoch's posteriorgram.
"""

import argparse
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

_RECIPE_DIR = Path(__file__).resolve().parent
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

import train as baseline_train
from conformer import Conformer as BaseConformer
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.utils import AttributeDict, MetricsTracker, encode_supervisions


def apply_label_prior(
    log_probs: Tensor,
    label_prior: Tensor,
    alpha: float,
    floor: float,
    enabled: bool = True,
) -> Tensor:
    if alpha == 0.0 or not enabled:
        return log_probs
    prior = label_prior.to(device=log_probs.device, dtype=torch.float32)
    prior = prior.clamp_min(floor)
    prior = prior / prior.sum()
    return log_probs - float(alpha) * prior.log().to(log_probs.dtype).view(1, 1, -1)


class LabelPriorConformer(BaseConformer):
    def __init__(self, *args, **kwargs) -> None:
        num_classes = int(kwargs["num_classes"])
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "label_prior",
            torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32),
        )
        self.register_buffer(
            "label_prior_sum",
            torch.zeros(num_classes, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "label_prior_total", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer("label_prior_ready", torch.tensor(False))

    def prior_adjusted_log_probs(
        self, log_probs: Tensor, alpha: float, floor: float, enabled: bool
    ) -> Tensor:
        return apply_label_prior(
            log_probs,
            self.label_prior,
            alpha=alpha,
            floor=floor,
            enabled=enabled and bool(self.label_prior_ready.item()),
        )

    @torch.no_grad()
    def reset_label_prior_stats(self) -> None:
        self.label_prior_sum.zero_()
        self.label_prior_total.zero_()

    @torch.no_grad()
    def accumulate_label_prior_stats(
        self, log_probs: Tensor, supervision_segments: Tensor
    ) -> None:
        probs = log_probs.detach().float().exp()
        total_frames = 0
        for sequence_idx, start_frame, num_frames in supervision_segments.tolist():
            start = int(start_frame)
            end = start + int(num_frames)
            if end <= start:
                continue
            segment = probs[int(sequence_idx), start:end]
            self.label_prior_sum.add_(segment.sum(dim=0))
            total_frames += segment.size(0)
        if total_frames:
            self.label_prior_total.add_(float(total_frames))

    @torch.no_grad()
    def sync_and_update_label_prior(
        self, momentum: float, floor: float
    ) -> Dict[str, float]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.label_prior_sum)
            torch.distributed.all_reduce(self.label_prior_total)
        total = float(self.label_prior_total.item())
        if total > 0.0:
            new_prior = self.label_prior_sum / self.label_prior_total.clamp_min(1.0)
            new_prior = new_prior.clamp_min(floor)
            new_prior = new_prior / new_prior.sum()
            if momentum > 0.0:
                new_prior = momentum * self.label_prior.float() + (1.0 - momentum) * new_prior
                new_prior = new_prior.clamp_min(floor)
                new_prior = new_prior / new_prior.sum()
            self.label_prior.copy_(new_prior.to(self.label_prior.dtype))
            self.label_prior_ready.fill_(True)
        prior = self.label_prior.float().clamp_min(floor)
        prior = prior / prior.sum()
        stats = {
            "frames": total,
            "ready": float(self.label_prior_ready.item()),
            "blank": float(prior[0]),
            "min": float(prior.min()),
            "max": float(prior.max()),
            "entropy": float(-(prior * prior.log()).sum()),
        }
        self.reset_label_prior_stats()
        return stats


_BASELINE_TRAIN_ONE_EPOCH = baseline_train.train_one_epoch


def get_parser() -> argparse.ArgumentParser:
    parser = baseline_train.get_parser()
    parser.set_defaults(exp_dir="conformer_ctc2/exp_label_prior_timit")
    parser.add_argument("--label-prior-alpha", type=float, default=0.3)
    parser.add_argument("--label-prior-floor", type=float, default=math.exp(-12.0))
    parser.add_argument("--label-prior-momentum", type=float, default=0.0)
    parser.add_argument("--label-prior-start-epoch", type=int, default=2)
    parser.add_argument(
        "--label-prior-update-until-epoch",
        type=int,
        default=0,
        help="0 updates every epoch; otherwise this is the last update epoch.",
    )
    return parser


def _model_ref(model: Union[nn.Module, DDP]) -> LabelPriorConformer:
    return model.module if isinstance(model, DDP) else model


def _should_update(params: AttributeDict) -> bool:
    update_until = int(params.label_prior_update_until_epoch)
    return update_until <= 0 or int(params.cur_epoch) <= update_until


def _is_active(params: AttributeDict, model: LabelPriorConformer) -> bool:
    return (
        float(params.label_prior_alpha) > 0.0
        and bool(model.label_prior_ready.item())
        and int(params.cur_epoch) >= int(params.label_prior_start_epoch)
    )


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    graph_compiler: CtcTrainingGraphCompiler,
    is_training: bool,
    warmup: float = 1.0,
) -> Tuple[Tensor, MetricsTracker]:
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    with torch.set_grad_enabled(is_training):
        base_log_probs, _, _ = model(feature, supervisions, warmup=warmup)

    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )
    decoding_graph = graph_compiler.compile(texts)
    model_ref = _model_ref(model)
    should_update = _should_update(params)
    if is_training and should_update:
        model_ref.accumulate_label_prior_stats(base_log_probs, supervision_segments)
    active = _is_active(params, model_ref)
    adjusted = model_ref.prior_adjusted_log_probs(
        base_log_probs,
        alpha=params.label_prior_alpha,
        floor=params.label_prior_floor,
        enabled=active,
    )
    dense_fsa_vec = k2.DenseFsaVec(
        adjusted,
        supervision_segments,
        allow_truncate=params.subsampling_factor - 1,
    )
    ctc_loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    )
    if params.att_rate != 0.0:
        raise ValueError("TIMIT label-prior experiment is pure CTC")
    loss = ctc_loss
    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_frames = (feature_lens // params.subsampling_factor).sum().item()
    info["frames"] = num_frames
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    info["loss"] = loss.detach().cpu().item()
    info["label_prior_alpha"] = float(params.label_prior_alpha) * num_frames
    info["label_prior_blank"] = float(model_ref.label_prior[0]) * num_frames
    info["label_prior_active"] = float(active) * num_frames
    info["label_prior_update"] = float(should_update) * num_frames
    info["utterances"] = feature.size(0)
    info["utt_duration"] = feature_lens.sum().item()
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )
    return loss, info


def train_one_epoch(*args, **kwargs) -> None:
    params = kwargs["params"]
    model_ref = _model_ref(kwargs["model"])
    model_ref.reset_label_prior_stats()
    _BASELINE_TRAIN_ONE_EPOCH(*args, **kwargs)
    if _should_update(params):
        stats = model_ref.sync_and_update_label_prior(
            momentum=float(params.label_prior_momentum),
            floor=float(params.label_prior_floor),
        )
        action = "Updated"
    else:
        prior = model_ref.label_prior.float().clamp_min(params.label_prior_floor)
        prior = prior / prior.sum()
        stats = {
            "frames": 0.0,
            "ready": float(model_ref.label_prior_ready.item()),
            "blank": float(prior[0]),
            "min": float(prior.min()),
            "max": float(prior.max()),
            "entropy": float(-(prior * prior.log()).sum()),
        }
        model_ref.reset_label_prior_stats()
        action = "Kept fixed"
    model_avg = kwargs.get("model_avg")
    if model_avg is not None and hasattr(model_avg, "label_prior"):
        model_avg.label_prior.copy_(model_ref.label_prior.to(model_avg.label_prior))
        model_avg.label_prior_ready.copy_(
            model_ref.label_prior_ready.to(model_avg.label_prior_ready)
        )
    if kwargs.get("rank", 0) == 0:
        logging.info(
            "%s label prior after epoch %s: frames=%.0f ready=%.0f blank=%.6g "
            "min=%.6g max=%.6g entropy=%.6g",
            action,
            params.cur_epoch,
            stats["frames"],
            stats["ready"],
            stats["blank"],
            stats["min"],
            stats["max"],
            stats["entropy"],
        )


def run(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if args.label_prior_alpha < 0.0:
        raise ValueError("--label-prior-alpha must be non-negative")
    if args.label_prior_floor <= 0.0:
        raise ValueError("--label-prior-floor must be positive")
    if not 0.0 <= args.label_prior_momentum < 1.0:
        raise ValueError("--label-prior-momentum must be in [0, 1)")
    if args.label_prior_start_epoch < 1:
        raise ValueError("--label-prior-start-epoch must be >= 1")
    baseline_train.Conformer = LabelPriorConformer
    baseline_train.compute_loss = compute_loss
    baseline_train.train_one_epoch = train_one_epoch
    baseline_train.run(rank=rank, world_size=world_size, args=args)


def main() -> None:
    parser = get_parser()
    baseline_train.TimitAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    if args.world_size > 1:
        mp.spawn(run, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
