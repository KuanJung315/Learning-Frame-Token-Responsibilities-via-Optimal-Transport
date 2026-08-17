#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import (
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

from asr_datamodule import TimitAsrDataModule
from blank_gate_v2 import BlankGateHeadV2, BlankPriorHeadV2
from conformer import Conformer
from phone_graph_compiler import PhoneCtcTrainingGraphCompiler
from varctc_v2_utils import build_gated_log_probs_v2, encoder_lens_from_mask

from icefall.lexicon import Lexicon
from icefall.utils import AttributeDict


IGNORED_CHECKPOINT_KEYS = {
    "model",
    "model_avg",
    "optimizer",
    "scheduler",
    "grad_scaler",
    "sampler",
}


@dataclass
class UtteranceOutput:
    log_probs: torch.Tensor
    output_len: int
    log_p_nonblank: Optional[torch.Tensor] = None
    alpha_prior: Optional[torch.Tensor] = None
    alpha_post: Optional[torch.Tensor] = None


class TimitVIV2ForEval(nn.Module):
    """VarCTC-v2 wrapper whose module names match TIMIT training checkpoints."""

    def __init__(
        self,
        encoder: Conformer,
        ctc_head: nn.Linear,
        blank_gate: BlankGateHeadV2,
        blank_prior: BlankPriorHeadV2,
        prior_logit_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.ctc_head = ctc_head
        self.blank_gate = blank_gate
        self.blank_prior = blank_prior
        self.prior_logit_bias = float(prior_logit_bias)

    def forward_details(
        self,
        x: torch.Tensor,
        supervisions: Dict[str, Any],
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        gate: str = "prior",
        warmup: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        memory, memory_key_padding_mask = self.encoder.run_encoder(
            x, supervisions, warmup=warmup
        )
        encoder_out = memory.permute(1, 0, 2)
        encoder_out_lens = encoder_lens_from_mask(
            memory_key_padding_mask,
            batch_size=encoder_out.size(0),
            max_len=encoder_out.size(1),
            device=encoder_out.device,
        )

        log_p_nonblank = F.log_softmax(self.ctc_head(encoder_out), dim=-1)
        alpha_prior = self.blank_prior(encoder_out, encoder_out_lens)
        if self.prior_logit_bias != 0.0:
            alpha_prior = alpha_prior.float().clamp(1.0e-5, 1.0 - 1.0e-5)
            alpha_prior = torch.sigmoid(
                torch.logit(alpha_prior) + self.prior_logit_bias
            )

        alpha_post = None
        if targets is not None and target_lengths is not None:
            alpha_post = self.blank_gate(
                encoder_out,
                targets,
                target_lengths,
                encoder_out_lens,
            )

        if gate == "prior":
            alpha = alpha_prior
        elif gate == "posterior":
            if alpha_post is None:
                raise ValueError("posterior gate requires targets and target_lengths")
            alpha = alpha_post
        else:
            raise ValueError(f"Unsupported VI gate: {gate}")

        return {
            "log_probs": build_gated_log_probs_v2(log_p_nonblank, alpha),
            "log_p_nonblank": log_p_nonblank,
            "alpha_prior": alpha_prior,
            "alpha_post": alpha_post,
            "output_lens": encoder_out_lens,
            "encoder_memory": memory,
            "memory_key_padding_mask": memory_key_padding_mask,
        }

    def forward(self, x, supervisions=None, warmup: float = 1.0):
        details = self.forward_details(x, supervisions, gate="prior", warmup=warmup)
        return (
            details["log_probs"],
            details["encoder_memory"],
            details["memory_key_padding_mask"],
        )


def checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {k: v for k, v in checkpoint.items() if k not in IGNORED_CHECKPOINT_KEYS}


def params_from_checkpoint(checkpoint_path: Path) -> AttributeDict:
    return AttributeDict(checkpoint_metadata(checkpoint_path))


def _load_model_state(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)


def build_phone_graph(
    lang_dir: Path,
    device: torch.device,
) -> Tuple[Lexicon, PhoneCtcTrainingGraphCompiler]:
    lexicon = Lexicon(lang_dir)
    graph = PhoneCtcTrainingGraphCompiler(lexicon=lexicon, device=device)
    return lexicon, graph


def build_baseline_model(
    checkpoint_path: Path,
    lang_dir: Path,
    device: torch.device,
) -> Tuple[Conformer, AttributeDict]:
    params = params_from_checkpoint(checkpoint_path)
    lexicon = Lexicon(lang_dir)
    num_classes = max(lexicon.tokens) + 1
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )
    _load_model_state(model, checkpoint_path)
    model.to(device).eval()
    return model, params


def build_vi_model(
    checkpoint_path: Path,
    lang_dir: Path,
    device: torch.device,
    prior_logit_bias: float = 0.0,
) -> Tuple[TimitVIV2ForEval, AttributeDict]:
    params = params_from_checkpoint(checkpoint_path)
    lexicon = Lexicon(lang_dir)
    num_classes = max(lexicon.tokens) + 1
    encoder = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )
    ctc_head = nn.Linear(params.encoder_dim, num_classes - 1)
    blank_gate = BlankGateHeadV2(
        d_model=params.encoder_dim,
        vocab_size=num_classes,
        d_attn=params.label_embed_dim,
        init_blank_prob=params.init_blank_prob,
    )
    blank_prior = BlankPriorHeadV2(
        d_model=params.encoder_dim,
        init_blank_prob=params.init_blank_prob,
    )
    model = TimitVIV2ForEval(
        encoder=encoder,
        ctc_head=ctc_head,
        blank_gate=blank_gate,
        blank_prior=blank_prior,
        prior_logit_bias=prior_logit_bias,
    )
    _load_model_state(model, checkpoint_path)
    model.to(device).eval()
    return model, params


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    TimitAsrDataModule.add_arguments(parser)
    parser.set_defaults(
        valid_cuts_name="timit_cuts_DEV_phone_nosa.jsonl.gz",
        test_cuts_name="timit_cuts_TEST_phone_nosa.jsonl.gz",
        return_cuts=True,
        shuffle=False,
        drop_last=False,
        enable_spec_aug=False,
        enable_musan=False,
        num_workers=0,
    )


def get_split_cuts(datamodule: TimitAsrDataModule, split: str, max_cuts: int) -> CutSet:
    if split == "dev":
        cuts = datamodule.valid_cuts()
    elif split == "test":
        cuts = datamodule.test_cuts()
    else:
        raise ValueError(f"Unsupported split: {split}")
    if max_cuts > 0:
        cuts = cuts.subset(first=max_cuts)
    return cuts


def get_split_dataloader(args: argparse.Namespace, split: str):
    datamodule = TimitAsrDataModule(args)
    cuts = get_split_cuts(datamodule, split=split, max_cuts=args.max_cuts)
    input_strategy = (
        OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        if args.on_the_fly_feats
        else PrecomputedFeatures()
    )
    dataset = K2SpeechRecognitionDataset(
        input_strategy=input_strategy,
        return_cuts=True,
    )
    sampler = SimpleCutSampler(
        cuts,
        max_duration=args.max_duration,
        shuffle=False,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=0,
        persistent_workers=False,
    )


def batch_token_ids_in_input_order(
    batch: Dict[str, Any],
    graph: PhoneCtcTrainingGraphCompiler,
    device: torch.device,
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    texts: Sequence[str] = batch["supervisions"]["text"]
    sequence_idx = batch["supervisions"]["sequence_idx"].tolist()
    supervision_ids = graph.texts_to_ids(list(texts))
    batch_size = int(batch["inputs"].size(0))
    input_ids: List[List[int]] = [[] for _ in range(batch_size)]
    for sup_idx, seq_idx in enumerate(sequence_idx):
        input_ids[int(seq_idx)] = supervision_ids[sup_idx]

    lengths = torch.tensor(
        [len(ids) for ids in input_ids], dtype=torch.long, device=device
    )
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    targets = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(input_ids):
        if ids:
            targets[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return supervision_ids, targets, lengths


@torch.no_grad()
def baseline_batch_outputs(
    model: Conformer,
    batch: Dict[str, Any],
    device: torch.device,
) -> List[UtteranceOutput]:
    features = batch["inputs"].to(device)
    log_probs, _, memory_key_padding_mask = model(
        features, batch["supervisions"], warmup=1.0
    )
    output_lens = encoder_lens_from_mask(
        memory_key_padding_mask,
        batch_size=log_probs.size(0),
        max_len=log_probs.size(1),
        device=log_probs.device,
    )
    outputs = []
    for i, length in enumerate(output_lens.tolist()):
        length = max(int(length), 1)
        outputs.append(
            UtteranceOutput(
                log_probs=log_probs[i, :length].detach().cpu(),
                output_len=length,
            )
        )
    return outputs


@torch.no_grad()
def vi_batch_outputs(
    model: TimitVIV2ForEval,
    batch: Dict[str, Any],
    graph: PhoneCtcTrainingGraphCompiler,
    device: torch.device,
    gate: str,
) -> Tuple[List[UtteranceOutput], List[List[int]]]:
    supervision_ids, targets, target_lengths = batch_token_ids_in_input_order(
        batch, graph=graph, device=device
    )
    features = batch["inputs"].to(device)
    details = model.forward_details(
        features,
        batch["supervisions"],
        targets=targets,
        target_lengths=target_lengths,
        gate=gate,
    )
    outputs = []
    for i, length in enumerate(details["output_lens"].tolist()):
        length = max(int(length), 1)
        alpha_post = details["alpha_post"]
        outputs.append(
            UtteranceOutput(
                log_probs=details["log_probs"][i, :length].detach().cpu(),
                output_len=length,
                log_p_nonblank=details["log_p_nonblank"][i, :length].detach().cpu(),
                alpha_prior=details["alpha_prior"][i, :length].detach().cpu(),
                alpha_post=(
                    alpha_post[i, :length].detach().cpu()
                    if alpha_post is not None
                    else None
                ),
            )
        )
    return outputs, supervision_ids


def greedy_runs(
    log_probs: torch.Tensor,
    blank_id: int = 0,
) -> List[Dict[str, Any]]:
    ids = log_probs.argmax(dim=-1).tolist()
    runs: List[Dict[str, Any]] = []
    start = 0
    while start < len(ids):
        end = start + 1
        while end < len(ids) and ids[end] == ids[start]:
            end += 1
        token_id = int(ids[start])
        if token_id != blank_id:
            runs.append(
                {
                    "token_id": token_id,
                    "start_frame": start,
                    "end_frame": end,
                    "center_frame": 0.5 * (start + end - 1),
                }
            )
        start = end
    return runs


def token_ids_to_symbols(
    token_ids: Sequence[int],
    token_table,
) -> List[str]:
    symbols = []
    for token_id in token_ids:
        symbol = token_table[int(token_id)]
        if symbol and symbol != "<eps>" and not symbol.startswith("#"):
            symbols.append(symbol)
    return symbols


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)
