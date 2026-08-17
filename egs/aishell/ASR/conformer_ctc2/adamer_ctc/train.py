#!/usr/bin/env python3
"""Train AISHELL conformer_ctc2 with adaptive maximum-entropy CTC regularization.

This is the AISHELL (char-based) counterpart of the LibriSpeech adamer_ctc
overlay. It reuses the existing AISHELL conformer_ctc2 model, data pipeline,
optimizer, checkpointing, and decoder unchanged, and only swaps in the
AdaMER Conformer and the AdaMER compute_loss.

The exact CTC path entropy requires a single token sequence per utterance.
This is well-defined for char lang dirs (each text maps to a unique char-id
sequence), exactly as for BPE. Phone lang dirs are rejected because a single
transcript may admit multiple pronunciations.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_ASR_DIR = _RECIPE_DIR.parent
for _path in (_RECIPE_DIR, _ASR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train as baseline_train  # noqa: E402
from adamer_ctc.loss import ctc_path_entropy  # noqa: E402
from adamer_ctc.model import AdaMERConformer, set_initial_beta  # noqa: E402
from adamer_ctc.objective import adamer_objectives  # noqa: E402
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler  # noqa: E402
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler  # noqa: E402
from icefall.graph_compiler import CtcTrainingGraphCompiler  # noqa: E402
from icefall.utils import (  # noqa: E402
    AttributeDict,
    MetricsTracker,
    encode_supervisions,
)


def get_parser() -> argparse.ArgumentParser:
    parser = baseline_train.get_parser()
    parser.add_argument(
        "--adamer-initial-beta",
        type=float,
        default=0.2,
        help="Initial adaptive entropy weight. The paper uses 0.2.",
    )
    parser.add_argument(
        "--adamer-target-entropy-per-token",
        type=float,
        default=1.1,
        help="Target path entropy multiplier. The paper uses H_target=1.1*U.",
    )
    return parser


def _prepare_path_entropy_inputs(
    log_probs: Tensor,
    supervision_segments: Tensor,
    token_ids: List[List[int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build padded, supervision-sorted tensors for the entropy DP."""
    log_prob_segments = []
    input_lengths = []
    for sequence_idx, start_frame, num_frames in supervision_segments.tolist():
        segment = log_probs[
            int(sequence_idx),
            int(start_frame) : int(start_frame + num_frames),
        ]
        log_prob_segments.append(segment)
        input_lengths.append(segment.size(0))

    padded_log_probs = pad_sequence(
        log_prob_segments,
        batch_first=False,
        padding_value=0.0,
    ).float()
    input_lengths_tensor = torch.tensor(
        input_lengths, device=log_probs.device, dtype=torch.long
    )
    target_lengths = torch.tensor(
        [len(ids) for ids in token_ids],
        device=log_probs.device,
        dtype=torch.long,
    )
    max_target_length = int(target_lengths.max().item()) if token_ids else 0
    targets = torch.zeros(
        len(token_ids),
        max_target_length,
        device=log_probs.device,
        dtype=torch.long,
    )
    for i, ids in enumerate(token_ids):
        if ids:
            targets[i, : len(ids)] = torch.tensor(
                ids, device=log_probs.device, dtype=torch.long
            )

    return padded_log_probs, input_lengths_tensor, targets, target_lengths


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    graph_compiler: Union[
        BpeCtcTrainingGraphCompiler, CharCtcTrainingGraphCompiler
    ],
    is_training: bool,
    warmup: float = 1.0,
) -> Tuple[Tensor, MetricsTracker]:
    """Compute hybrid AED/AdaMER-CTC loss with stop-gradient dual updates."""
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"].to(device)
    assert feature.ndim == 3

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    with torch.set_grad_enabled(is_training):
        nnet_output, encoder_memory, memory_mask = model(
            feature, supervisions, warmup=warmup
        )

    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    if isinstance(
        graph_compiler, (BpeCtcTrainingGraphCompiler, CharCtcTrainingGraphCompiler)
    ):
        # Works with a BPE or char model: each text maps to one token sequence.
        token_ids = graph_compiler.texts_to_ids(texts)
        decoding_graph = graph_compiler.compile(token_ids)
    elif isinstance(graph_compiler, CtcTrainingGraphCompiler):
        raise NotImplementedError(
            "AdaMER path entropy currently requires a BPE or char lang dir. "
            "Phone graphs may contain multiple pronunciations, so their exact "
            "path entropy cannot be recovered from a single token sequence."
        )
    else:
        raise ValueError(f"Unsupported type of graph compiler: {type(graph_compiler)}")

    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
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

    entropy_inputs = _prepare_path_entropy_inputs(
        nnet_output, supervision_segments, token_ids
    )
    path_log_likelihood, path_entropy = ctc_path_entropy(*entropy_inputs, blank=0)
    valid_entropy = (
        torch.isfinite(path_log_likelihood)
        & torch.isfinite(path_entropy)
        & (path_entropy >= -1.0e-4)
    )
    path_entropy = path_entropy.clamp_min(0.0).masked_fill(~valid_entropy, 0.0)

    model_ref = model.module if isinstance(model, DDP) else model
    beta = model_ref.adamer_beta()
    if not is_training:
        beta = beta.detach()
    target_entropy = (
        entropy_inputs[3].to(path_entropy.dtype)
        * float(params.adamer_target_entropy_per_token)
    )
    target_entropy = target_entropy.masked_fill(~valid_entropy, 0.0)

    adamer_ctc_loss, beta_loss, entropy_regularizer = adamer_objectives(
        ctc_loss=ctc_loss,
        path_entropy=path_entropy,
        target_entropy=target_entropy,
        beta=beta,
    )

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
        model_loss = (
            (1.0 - params.att_rate) * adamer_ctc_loss
            + params.att_rate * att_loss
        )
    else:
        model_loss = adamer_ctc_loss
        att_loss = nnet_output.new_tensor(0.0)

    # L_beta is needed only while training beta. Validation reports the model
    # objective, not the detached dual-controller update.
    loss = model_loss + beta_loss if is_training else model_loss
    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_frames = (feature_lens // params.subsampling_factor).sum().item()
    info["frames"] = num_frames
    num_utterances = feature.size(0)
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    # NOTE: MetricsTracker.__str__ reserves the "utt_" prefix for
    # utterance-normalized keys and only whitelists utt_duration /
    # utt_pad_proportion. All AdaMER diagnostics therefore use plain keys,
    # which MetricsTracker normalizes by frame count (like ctc_loss).
    info["entropy_reg"] = entropy_regularizer.detach().cpu().item()
    info["beta_loss"] = beta_loss.detach().cpu().item() if is_training else 0.0
    info["path_entropy"] = path_entropy.sum().detach().cpu().item()
    info["target_entropy"] = target_entropy.sum().detach().cpu().item()
    # Scale by num_frames so the frame-normalization in __str__ recovers the
    # (frame-weighted average) beta value itself.
    info["adamer_beta"] = beta.detach().cpu().item() * num_frames
    info["invalid_entropy"] = (~valid_entropy).sum().detach().cpu().item()
    if params.att_rate != 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()
    if is_training:
        info["loss"] = loss.detach().cpu().item()
    else:
        # Keep best-valid selection comparable across epochs as beta changes.
        base_valid_loss = (1.0 - params.att_rate) * ctc_loss
        if params.att_rate != 0.0:
            base_valid_loss = base_valid_loss + params.att_rate * att_loss
        info["loss"] = base_valid_loss.detach().cpu().item()
    info["utterances"] = num_utterances
    info["utt_duration"] = feature_lens.sum().item()
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )
    return loss, info


def run(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if args.adamer_initial_beta <= 0:
        raise ValueError("--adamer-initial-beta must be positive")
    if args.adamer_target_entropy_per_token < 0:
        raise ValueError("--adamer-target-entropy-per-token must be non-negative")

    set_initial_beta(args.adamer_initial_beta)
    baseline_train.Conformer = AdaMERConformer
    baseline_train.compute_loss = compute_loss
    baseline_train.run(rank=rank, world_size=world_size, args=args)


def main() -> None:
    parser = get_parser()
    baseline_train.AishellAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    if args.world_size > 1:
        mp.spawn(run, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
