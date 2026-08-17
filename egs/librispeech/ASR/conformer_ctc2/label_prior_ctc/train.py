#!/usr/bin/env python3
"""Train conformer_ctc2 with CTC label priors.

This implements the objective from "Less Peaky and More Accurate CTC Forced
Alignment by Label Priors": CTC path scores use y_k^t / P(k)^alpha, where the
unigram label prior P(k) is estimated from the previous epoch's posteriorgram.
"""

import argparse
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Tuple, Union

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

import train as baseline_train  # noqa: E402
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler  # noqa: E402
from icefall.graph_compiler import CtcTrainingGraphCompiler  # noqa: E402
from icefall.utils import AttributeDict, MetricsTracker, encode_supervisions  # noqa: E402
from label_prior_ctc.model import LabelPriorConformer  # noqa: E402


_BASELINE_TRAIN_ONE_EPOCH = baseline_train.train_one_epoch
GraphCompiler = Union[BpeCtcTrainingGraphCompiler, CtcTrainingGraphCompiler]


def get_parser() -> argparse.ArgumentParser:
    parser = baseline_train.get_parser()
    parser.set_defaults(exp_dir="conformer_ctc2/label_prior_ctc/exp_li100h")
    parser.add_argument(
        "--label-prior-alpha",
        type=float,
        default=0.3,
        help=(
            "Scale alpha in y_k^t / P(k)^alpha. 0 disables the prior and "
            "recovers standard CTC. The official paper uses 0.3 for the "
            "reported label-prior CTC system."
        ),
    )
    parser.add_argument(
        "--label-prior-floor",
        type=float,
        default=math.exp(-12.0),
        help=(
            "Numerical floor for estimated unigram label priors. Defaults to "
            "exp(-12), matching the official open-source implementation."
        ),
    )
    parser.add_argument(
        "--label-prior-momentum",
        type=float,
        default=0.0,
        help=(
            "EMA momentum for epoch-level prior updates. 0 means replace by "
            "the latest full-epoch posteriorgram estimate."
        ),
    )
    parser.add_argument(
        "--label-prior-start-epoch",
        type=int,
        default=2,
        help=(
            "First epoch that applies the label prior. The default preserves "
            "the original behavior: collect an epoch-1 posteriorgram, then "
            "apply it from epoch 2."
        ),
    )
    parser.add_argument(
        "--label-prior-update-until-epoch",
        type=int,
        default=0,
        help=(
            "Last epoch that updates the label prior. 0 means update after "
            "every epoch. Set this to start_epoch - 1 to estimate the prior "
            "during standard-CTC warmup and then keep it fixed."
        ),
    )
    return parser


def _model_ref(model: Union[nn.Module, DDP]) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _cur_epoch(params: AttributeDict) -> int:
    return int(getattr(params, "cur_epoch", 1))


def _should_update_label_prior(params: AttributeDict) -> bool:
    update_until = int(getattr(params, "label_prior_update_until_epoch", 0))
    return update_until <= 0 or _cur_epoch(params) <= update_until


def _label_prior_is_active(params: AttributeDict, model_ref: nn.Module) -> bool:
    start_epoch = int(getattr(params, "label_prior_start_epoch", 2))
    return (
        float(params.label_prior_alpha) > 0.0
        and bool(model_ref.label_prior_ready.item())
        and _cur_epoch(params) >= start_epoch
    )


def _label_prior_snapshot(
    model_ref: nn.Module,
    floor: float,
    frames: float = 0.0,
) -> dict:
    prior = model_ref.label_prior.detach().float().clamp_min(floor)
    prior = prior / prior.sum()
    entropy = -(prior * prior.log()).sum().item()
    return {
        "frames": float(frames),
        "ready": float(model_ref.label_prior_ready.item()),
        "blank": prior[0].item(),
        "min": prior.min().item(),
        "max": prior.max().item(),
        "entropy": entropy,
    }


def _copy_label_prior_to_model_avg(model_ref: nn.Module, model_avg: nn.Module) -> None:
    if model_avg is None or not hasattr(model_avg, "label_prior"):
        return

    with torch.no_grad():
        model_avg.label_prior.copy_(
            model_ref.label_prior.detach().to(model_avg.label_prior.device)
        )
        if hasattr(model_avg, "label_prior_ready"):
            model_avg.label_prior_ready.copy_(
                model_ref.label_prior_ready.detach().to(
                    model_avg.label_prior_ready.device
                )
            )


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    graph_compiler: GraphCompiler,
    is_training: bool,
    warmup: float = 1.0,
) -> Tuple[Tensor, MetricsTracker]:
    """Compute hybrid AED/label-prior-CTC loss."""
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"].to(device)
    assert feature.ndim == 3

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    with torch.set_grad_enabled(is_training):
        base_log_probs, encoder_memory, memory_mask = model(
            feature, supervisions, warmup=warmup
        )

    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
        token_ids = graph_compiler.texts_to_ids(texts)
        decoding_graph = graph_compiler.compile(token_ids)
    elif isinstance(graph_compiler, CtcTrainingGraphCompiler):
        decoding_graph = graph_compiler.compile(texts)
    else:
        raise ValueError(f"Unsupported type of graph compiler: {type(graph_compiler)}")

    model_ref = _model_ref(model)
    should_update_prior = _should_update_label_prior(params)
    if is_training and should_update_prior:
        model_ref.accumulate_label_prior_stats(base_log_probs, supervision_segments)

    prior_active = _label_prior_is_active(params, model_ref)
    prior_log_probs = model_ref.prior_adjusted_log_probs(
        base_log_probs,
        alpha=params.label_prior_alpha,
        floor=params.label_prior_floor,
        enabled=prior_active,
    )
    dense_fsa_vec = k2.DenseFsaVec(
        prior_log_probs,
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
        if not isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
            raise ValueError("Attention decoder training requires a BPE lang dir")
        with torch.set_grad_enabled(is_training):
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = model_ref.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        loss = (1.0 - params.att_rate) * ctc_loss + params.att_rate * att_loss
    else:
        loss = ctc_loss
        att_loss = prior_log_probs.new_tensor(0.0)

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_frames = (feature_lens // params.subsampling_factor).sum().item()
    info["frames"] = num_frames
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    if params.att_rate != 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()
    info["loss"] = loss.detach().cpu().item()
    info["label_prior_alpha"] = float(params.label_prior_alpha) * num_frames
    info["label_prior_blank"] = (
        model_ref.label_prior[0].detach().cpu().item() * num_frames
    )
    info["label_prior_max"] = (
        model_ref.label_prior.detach().float().max().cpu().item() * num_frames
    )
    info["label_prior_active"] = float(prior_active) * num_frames
    info["label_prior_update"] = float(should_update_prior) * num_frames
    info["utterances"] = feature.size(0)
    info["utt_duration"] = feature_lens.sum().item()
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    return loss, info


def train_one_epoch(*args, **kwargs) -> None:
    """Run baseline training for one epoch, then update the label prior."""
    params = kwargs["params"]
    model = kwargs["model"]
    model_avg = kwargs.get("model_avg", None)
    tb_writer = kwargs.get("tb_writer", None)
    rank = kwargs.get("rank", 0)

    model_ref = _model_ref(model)
    model_ref.reset_label_prior_stats()
    _BASELINE_TRAIN_ONE_EPOCH(*args, **kwargs)

    should_update_prior = _should_update_label_prior(params)
    if should_update_prior:
        stats = model_ref.sync_and_update_label_prior(
            momentum=float(params.label_prior_momentum),
            floor=float(params.label_prior_floor),
        )
        log_action = "Updated"
    else:
        stats = _label_prior_snapshot(
            model_ref,
            floor=float(params.label_prior_floor),
        )
        model_ref.reset_label_prior_stats()
        log_action = "Kept fixed"

    _copy_label_prior_to_model_avg(model_ref, model_avg)

    if (
        int(getattr(params, "valid_interval_batches", 0)) > 0
        and not bool(getattr(params, "profile_cuda_memory", False))
    ):
        # Validate after updating the prior so best-valid-loss.pt contains the
        # exact label-prior state that produced the validation score.
        valid_info = baseline_train.compute_validation_loss(
            params=params,
            model=model,
            graph_compiler=kwargs["graph_compiler"],
            valid_dl=kwargs["valid_dl"],
            world_size=kwargs.get("world_size", 1),
        )
        model.train()
        logging.info(
            "Epoch %s, post-prior epoch-end validation: %s",
            params.cur_epoch,
            valid_info,
        )
        if tb_writer is not None:
            valid_info.write_summary(
                tb_writer, "train/valid_epoch_end_", params.batch_idx_train
            )

    if rank == 0:
        logging.info(
            "%s label prior after epoch %s: frames=%.0f ready=%.0f blank=%.6g "
            "min=%.6g max=%.6g entropy=%.6g",
            log_action,
            params.cur_epoch,
            stats["frames"],
            stats["ready"],
            stats["blank"],
            stats["min"],
            stats["max"],
            stats["entropy"],
        )
        if tb_writer is not None:
            step = params.batch_idx_train
            tb_writer.add_scalar("train/label_prior_blank", stats["blank"], step)
            tb_writer.add_scalar("train/label_prior_min", stats["min"], step)
            tb_writer.add_scalar("train/label_prior_max", stats["max"], step)
            tb_writer.add_scalar("train/label_prior_entropy", stats["entropy"], step)


def run(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if args.label_prior_alpha < 0.0:
        raise ValueError("--label-prior-alpha must be non-negative")
    if args.label_prior_floor <= 0.0:
        raise ValueError("--label-prior-floor must be positive")
    if not (0.0 <= args.label_prior_momentum < 1.0):
        raise ValueError("--label-prior-momentum must be in [0, 1)")
    if args.label_prior_start_epoch < 1:
        raise ValueError("--label-prior-start-epoch must be >= 1")
    if args.label_prior_update_until_epoch < 0:
        raise ValueError("--label-prior-update-until-epoch must be >= 0")

    baseline_train.Conformer = LabelPriorConformer
    baseline_train.compute_loss = compute_loss
    baseline_train.train_one_epoch = train_one_epoch
    baseline_train.run(rank=rank, world_size=world_size, args=args)


def main() -> None:
    parser = get_parser()
    baseline_train.LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    if args.world_size > 1:
        mp.spawn(run, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
