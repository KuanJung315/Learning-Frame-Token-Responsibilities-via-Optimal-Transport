#!/usr/bin/env python3
"""Train architecture-controlled 5M TDNN standard/Label-Prior phone CTC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.multiprocessing as mp

SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR.parent
ASR_DIR = RECIPE_DIR.parent
for path in (RECIPE_DIR, ASR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train as baseline_train  # noqa: E402
import label_prior_ctc.train as label_prior_train  # noqa: E402
from tdnn_alignment.model import TdnnLabelPriorCTC  # noqa: E402
from word_phone_graph_compiler import (  # noqa: E402
    WordPhoneCtcTrainingGraphCompiler,
)


_BASE_GET_PARAMS = baseline_train.get_params


class ProtocolPhoneGraphCompiler(WordPhoneCtcTrainingGraphCompiler):
    """Constructor/compile adapter for the baseline trainer."""

    lang_dir: Path

    def __init__(self, lexicon, device) -> None:
        super().__init__(self.lang_dir, lexicon=lexicon, device=device)

    def compile(self, transcripts, modified: bool = False):
        if transcripts and isinstance(transcripts[0], str):
            transcripts = self.texts_to_ids(transcripts)
        return super().compile(transcripts, modified=modified)


def get_parser() -> argparse.ArgumentParser:
    parser = label_prior_train.get_parser()
    parser.set_defaults(
        exp_dir="conformer_ctc2/tdnn_alignment/exp_li100h_phone_lp0p3_seed42",
        lang_dir="data/lang_phone_nostress",
        att_rate=0.0,
        num_decoder_layers=0,
        # Label-prior statistics should cover every retained training frame.
        drop_last=False,
    )
    return parser


def _tdnn_params():
    params = _BASE_GET_PARAMS()
    params.subsampling_factor = 2
    params.encoder_dim = 640
    params.num_encoder_layers = 3
    params.architecture = "tdnn640_s2_k5_k3_k3_ffn5"
    # The phone/input-frame ratio reaches about 0.183 on the prepared Libri
    # manifests.  0.2 retains every stride-2 CTC-feasible cut while the base
    # recipe keeps its historical 0.1 default for BPE experiments.
    params.ctc_filter_max_input_ratio = 0.2
    return params


def run(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if args.att_rate != 0.0 or args.num_decoder_layers != 0:
        raise ValueError("The 5M alignment control requires pure CTC")
    if "lang_phone" not in str(args.lang_dir):
        raise ValueError("The controlled comparison requires a phone lang dir")
    if args.label_prior_alpha < 0.0:
        raise ValueError("--label-prior-alpha must be non-negative")

    ProtocolPhoneGraphCompiler.lang_dir = Path(args.lang_dir)
    baseline_train.get_params = _tdnn_params
    baseline_train.Conformer = TdnnLabelPriorCTC
    baseline_train.CtcTrainingGraphCompiler = ProtocolPhoneGraphCompiler
    baseline_train.compute_loss = label_prior_train.compute_loss
    baseline_train.train_one_epoch = label_prior_train.train_one_epoch
    label_prior_train.CtcTrainingGraphCompiler = ProtocolPhoneGraphCompiler
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
