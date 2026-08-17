#!/usr/bin/env python3
"""Train the baseline Conformer with matched CR-CTC regularization.

This control intentionally matches the CR branch used by cr_ctc_ours: view 1
is the regular dataloader-augmented input and view 2 adds the same extra masks.
It isolates the contribution of VFTA without changing the augmentation recipe.
"""

import argparse
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
from cr_ctc_utils import (  # noqa: E402
    cr_ctc_consistency_loss,
    make_cr_view,
)
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler  # noqa: E402
from icefall.graph_compiler import CtcTrainingGraphCompiler  # noqa: E402
from icefall.utils import AttributeDict, MetricsTracker, encode_supervisions  # noqa: E402


GraphCompiler = Union[BpeCtcTrainingGraphCompiler, CtcTrainingGraphCompiler]


def get_parser() -> argparse.ArgumentParser:
    parser = baseline_train.get_parser()
    parser.set_defaults(exp_dir="conformer_ctc2/cr_ctc/exp_li100h")
    parser.add_argument(
        "--cr-ctc-weight",
        type=float,
        default=0.1,
        help="Weight for the frame-distribution consistency loss.",
    )
    parser.add_argument(
        "--cr-ctc-supervised-weight",
        type=float,
        default=1.0,
        help="Second-view CTC weight before normalizing the two-view average.",
    )
    parser.add_argument(
        "--cr-stop-gradient",
        type=baseline_train.str2bool,
        default=True,
        help="Use symmetric stop-gradient KL between the two views.",
    )
    parser.add_argument("--cr-temperature", type=float, default=1.0)
    parser.add_argument("--cr-num-time-masks", type=int, default=2)
    parser.add_argument("--cr-time-mask-size", type=int, default=80)
    parser.add_argument("--cr-num-feature-masks", type=int, default=2)
    parser.add_argument("--cr-feature-mask-size", type=int, default=27)
    parser.add_argument("--cr-mask-value", type=float, default=0.0)
    return parser


def _output_lengths(
    log_probs: Tensor,
    memory_mask: Tensor,
) -> Tensor:
    if memory_mask is None:
        return torch.full(
            (log_probs.size(0),),
            log_probs.size(1),
            dtype=torch.long,
            device=log_probs.device,
        )
    return (~memory_mask).sum(dim=1).to(dtype=torch.long)


def _ctc_loss(
    params: AttributeDict,
    log_probs: Tensor,
    supervision_segments: Tensor,
    decoding_graph: k2.Fsa,
) -> Tensor:
    dense_fsa_vec = k2.DenseFsaVec(
        log_probs,
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
) -> Tuple[Tensor, MetricsTracker]:
    """Compute the matched two-view CR-CTC/AED objective."""
    model_ref = model.module if hasattr(model, "module") else model
    device = next(model_ref.parameters()).device
    feature = batch["inputs"].to(device)
    assert feature.ndim == 3

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
        token_ids = graph_compiler.texts_to_ids(texts)
        decoding_graph = graph_compiler.compile(token_ids)
    elif isinstance(graph_compiler, CtcTrainingGraphCompiler):
        decoding_graph = graph_compiler.compile(texts)
    else:
        raise ValueError(f"Unsupported graph compiler: {type(graph_compiler)}")

    use_cr = is_training and float(params.cr_ctc_weight) > 0.0
    with torch.set_grad_enabled(is_training):
        log_probs, encoder_memory, memory_mask = model(
            feature, supervisions, warmup=warmup
        )
        if use_cr:
            feature_cr = make_cr_view(feature, feature_lens, params)
            log_probs_cr, _, memory_mask_cr = model(
                feature_cr, supervisions, warmup=warmup
            )
        else:
            log_probs_cr = log_probs
            memory_mask_cr = memory_mask

    ctc_loss = _ctc_loss(
        params, log_probs, supervision_segments, decoding_graph
    )
    supervised_weight = (
        float(params.cr_ctc_supervised_weight) if is_training else 0.0
    )
    if use_cr and supervised_weight > 0.0:
        ctc_loss_cr = _ctc_loss(
            params, log_probs_cr, supervision_segments, decoding_graph
        )
        ctc_for_model = (
            ctc_loss + supervised_weight * ctc_loss_cr
        ) / (1.0 + supervised_weight)
    else:
        ctc_loss_cr = ctc_loss.new_zeros(())
        ctc_for_model = ctc_loss

    if not torch.isfinite(ctc_for_model):
        raise ValueError("CR-CTC loss is inf/nan; reduce max-duration.")

    if use_cr:
        cr_loss = cr_ctc_consistency_loss(
            log_probs_a=log_probs,
            log_probs_b=log_probs_cr,
            lengths_a=_output_lengths(log_probs, memory_mask),
            lengths_b=_output_lengths(log_probs_cr, memory_mask_cr),
            stop_gradient=bool(params.cr_stop_gradient),
            temperature=float(params.cr_temperature),
        )
    else:
        cr_loss = log_probs.new_zeros(log_probs.size(0))

    if params.att_rate != 0.0:
        with torch.set_grad_enabled(is_training):
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = model_ref.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        loss = (
            (1.0 - params.att_rate) * ctc_for_model
            + params.att_rate * att_loss
            + float(params.cr_ctc_weight) * cr_loss.sum()
        )
    else:
        att_loss = log_probs.new_tensor(0.0)
        loss = ctc_for_model + float(params.cr_ctc_weight) * cr_loss.sum()

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    info["ctc_loss_cr"] = ctc_loss_cr.detach().cpu().item()
    info["cr_ctc_loss"] = cr_loss.sum().detach().cpu().item()
    if params.att_rate != 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()
    info["loss"] = loss.detach().cpu().item()
    info["utterances"] = feature.size(0)
    info["utt_duration"] = feature_lens.sum().item()
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )
    return loss, info


def run(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if args.cr_ctc_weight < 0:
        raise ValueError("--cr-ctc-weight must be non-negative")
    if args.cr_ctc_supervised_weight < 0:
        raise ValueError("--cr-ctc-supervised-weight must be non-negative")
    if args.cr_temperature <= 0:
        raise ValueError("--cr-temperature must be positive")

    baseline_train.compute_loss = compute_loss
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
