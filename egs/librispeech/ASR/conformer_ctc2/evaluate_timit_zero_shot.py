#!/usr/bin/env python3
"""Evaluate a LibriSpeech-trained phone VFTA model on TIMIT without fine-tuning.

Primary PBE/WBE metrics use the best transcript-constrained CTC path, matching
the Label-Prior CTC paper's forced-alignment definition.  TIMIT sentence text
is expanded with the *same* deterministic LibriSpeech lexicon used in training;
human phone boundaries are read only after inference for scoring.

Since the local LibriSpeech lexicon and TIMIT use different phone inventories,
PBE is reported together with exact phone-sequence match coverage after
stripping CMU stress.  WBE is matched by orthographic word identity and is not
affected by phone aliases.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import kaldialign
import numpy as np
import torch
import torch.nn as nn
from lhotse import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset, PrecomputedFeatures
from lhotse.dataset.sampling import SimpleCutSampler
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
FGW_DIR = SCRIPT_DIR / "vfta_fgw"
for path in (SCRIPT_DIR, FGW_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blank_gate_v2 import BlankGateHeadV2, BlankPriorHeadV2
from conformer import Conformer
from ctc_plan_consistency import ctc_token_occupancy_batched
from ot_fgw import (
    DiagonalResidualPSDFrameMetric,
    LearnablePSDFrameMetric,
    vi_fgw_loss_v2,
)
from ot_prior_v2 import vi_ot_loss_v2
from train_vi_ot_v2 import ConformerVIV2
from tdnn_alignment.model import TdnnLabelPriorCTC
from varctc_v2_utils import build_gated_log_probs_v2
from word_phone_graph_compiler import (
    PhoneTranscript,
    WordPhoneCtcTrainingGraphCompiler,
)

from icefall.lexicon import Lexicon
from icefall.utils import AttributeDict


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--model-family",
        choices=["auto", "vfta", "tdnn"],
        default="auto",
        help="Infer the checkpoint architecture or force one explicitly.",
    )
    parser.add_argument(
        "--checkpoint-state",
        choices=["model", "model_avg"],
        default="model",
        help="State dict to evaluate from an epoch checkpoint.",
    )
    parser.add_argument(
        "--decode-label-prior-alpha",
        type=float,
        default=None,
        help=(
            "TDNN only: override the checkpoint's training alpha during "
            "constrained Viterbi. Standard CTC uses 0."
        ),
    )
    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=Path("data/lang_phone_nostress"),
        help="The exact LibriSpeech phone language directory used for training.",
    )
    parser.add_argument(
        "--timit-manifest-dir",
        type=Path,
        default=Path("../../timit/ASR/data/fbank"),
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["DEV", "TEST"], default=["DEV", "TEST"]
    )
    parser.add_argument(
        "--include-sa",
        action="store_true",
        help=(
            "Include the repeated TIMIT SA1/SA2 sentences. The primary protocol "
            "excludes them (DEV=400, TEST=192) to avoid repeated-sentence bias."
        ),
    )
    parser.add_argument(
        "--gate",
        choices=["prior", "posterior"],
        default="prior",
        help=(
            "Use prior for the architecture-matched comparison to ordinary CTC. "
            "Posterior is a legal transcript-conditioned VFTA diagnostic."
        ),
    )
    parser.add_argument("--prior-logit-bias", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--max-cuts", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Check transcript, OOV, and phone/word match coverage without a model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/timit_zero_shot"),
    )
    return parser


def _items(cut, tier: str):
    supervision = cut.supervisions[0]
    if not supervision.alignment or tier not in supervision.alignment:
        raise ValueError(f"{cut.id}: missing {tier} alignment")
    return [
        item
        for item in supervision.alignment[tier]
        if str(item.symbol).strip()
    ]


def _intervals(items) -> Tuple[List[str], List[float], List[float]]:
    symbols = [str(item.symbol) for item in items]
    starts = [float(item.start) for item in items]
    ends = [float(item.start + item.duration) for item in items]
    return symbols, starts, ends


def normalize_phone(symbol: str) -> str:
    # LibriSpeech lexicon uses stress-marked CMU phones; TIMIT manifests in
    # this workspace have already been reduced to a 39-phone-like inventory.
    symbol = symbol.strip().lower()
    while symbol and symbol[-1].isdigit():
        symbol = symbol[:-1]
    return symbol


def _exact_match_pairs(
    gold: Sequence[str], predicted: Sequence[str]
) -> List[Tuple[int, int]]:
    """Return (predicted_index, gold_index) for exact edit-alignment matches."""
    pairs: List[Tuple[int, int]] = []
    gold_idx = predicted_idx = 0
    for ref, hyp in kaldialign.align(gold, predicted, "<eps>"):
        if ref != "<eps>" and hyp != "<eps>" and ref == hyp:
            pairs.append((predicted_idx, gold_idx))
        if ref != "<eps>":
            gold_idx += 1
        if hyp != "<eps>":
            predicted_idx += 1
    return pairs


def _gold_words(
    compiler: WordPhoneCtcTrainingGraphCompiler, cut
) -> Tuple[List[str], List[float], List[float]]:
    items = _items(cut, "word")
    words: List[str] = []
    starts: List[float] = []
    ends: List[float] = []
    for item in items:
        normalized = compiler.normalize_words(str(item.symbol))
        if len(normalized) != 1:
            continue
        words.append(normalized[0])
        starts.append(float(item.start))
        ends.append(float(item.start + item.duration))
    return words, starts, ends


def _gold_phones(cut) -> Tuple[List[str], List[float], List[float]]:
    symbols, starts, ends = _intervals(_items(cut, "phone"))
    kept = [
        i for i, symbol in enumerate(symbols) if normalize_phone(symbol) != "sil"
    ]
    return (
        [normalize_phone(symbols[i]) for i in kept],
        [starts[i] for i in kept],
        [ends[i] for i in kept],
    )


def _ctc_viterbi_token_frames(
    log_probs: torch.Tensor, labels: Sequence[int], blank_id: int = 0
) -> List[Tuple[int, int]]:
    """Best-path inclusive frame span for every transcript token position."""
    if not labels:
        return []
    lp = log_probs.float()
    num_frames = lp.size(0)
    labels_t = torch.tensor(labels, dtype=torch.long, device=lp.device)
    expanded = torch.full(
        (2 * len(labels) + 1,), blank_id, dtype=torch.long, device=lp.device
    )
    expanded[1::2] = labels_t
    num_states = expanded.numel()
    neg = torch.finfo(lp.dtype).min

    score = lp.new_full((num_states,), neg)
    score[0] = lp[0, blank_id]
    score[1] = lp[0, expanded[1]]
    backpointers: List[torch.Tensor] = []

    state_index = torch.arange(num_states, device=lp.device)
    for frame in range(1, num_frames):
        candidates = torch.stack(
            [
                score,
                torch.cat([score.new_full((1,), neg), score[:-1]]),
                torch.cat([score.new_full((2,), neg), score[:-2]]),
            ],
            dim=0,
        )
        skip_ok = (state_index % 2 == 1) & (state_index >= 2)
        if num_states > 2:
            skip_ok[2:] &= expanded[2:] != expanded[:-2]
        candidates[2] = candidates[2].masked_fill(~skip_ok, neg)
        best_score, transition = candidates.max(dim=0)
        score = best_score + lp[frame, expanded]
        predecessor = state_index - transition
        backpointers.append(predecessor)

    terminal = torch.tensor(
        [num_states - 1, num_states - 2], device=lp.device
    )
    terminal_score, terminal_choice = score[terminal].max(dim=0)
    if not torch.isfinite(terminal_score):
        raise RuntimeError(
            f"No valid CTC path for {num_frames} frames and {len(labels)} labels"
        )
    state = int(terminal[terminal_choice].item())
    path = [state]
    for predecessor in reversed(backpointers):
        state = int(predecessor[state].item())
        path.append(state)
    path.reverse()

    spans: List[Tuple[int, int]] = []
    for token_index in range(len(labels)):
        token_state = 2 * token_index + 1
        frames = [i for i, state in enumerate(path) if state == token_state]
        if not frames:
            raise RuntimeError(f"Viterbi path skipped token position {token_index}")
        spans.append((frames[0], frames[-1]))
    return spans


def _frame_center_seconds(
    frame: float, subsampling_factor: int, model_family: str = "vfta"
) -> float:
    if model_family == "tdnn":
        if subsampling_factor != 2:
            raise ValueError("The reference TDNN timing requires stride 2")
        # Same-padding k=5/stride=2 is centred on input frame 2*t.
        return 0.005 + 0.01 * subsampling_factor * float(frame)
    if subsampling_factor != 4:
        raise ValueError(
            "This Conformer timing conversion is defined for Conv2d subsampling=4"
        )
    return 0.035 + 0.01 * subsampling_factor * float(frame)


def _frames_to_intervals(
    spans: Sequence[Tuple[int, int]],
    duration: float,
    subsampling_factor: int,
    model_family: str = "vfta",
) -> Tuple[List[float], List[float]]:
    half_step = 0.005 * subsampling_factor
    starts = [
        max(
            0.0,
            _frame_center_seconds(begin, subsampling_factor, model_family)
            - half_step,
        )
        for begin, _ in spans
    ]
    ends = [
        min(
            duration,
            _frame_center_seconds(end, subsampling_factor, model_family)
            + half_step,
        )
        for _, end in spans
    ]
    return starts, ends


def _word_intervals(
    transcript: PhoneTranscript,
    phone_starts: Sequence[float],
    phone_ends: Sequence[float],
) -> Tuple[List[float], List[float]]:
    starts = [phone_starts[begin] for begin, _ in transcript.word_phone_spans]
    ends = [phone_ends[end - 1] for _, end in transcript.word_phone_spans]
    return starts, ends


def _paired_errors(
    pairs: Sequence[Tuple[int, int]],
    pred_starts: Sequence[float],
    pred_ends: Sequence[float],
    gold_starts: Sequence[float],
    gold_ends: Sequence[float],
) -> Dict[str, object]:
    onset = [
        1000.0 * abs(pred_starts[pred] - gold_starts[gold])
        for pred, gold in pairs
    ]
    offset = [
        1000.0 * abs(pred_ends[pred] - gold_ends[gold])
        for pred, gold in pairs
    ]
    boundary = [(left + right) / 2.0 for left, right in zip(onset, offset)]
    return {"onset": onset, "offset": offset, "boundary": boundary}


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _column_normalize(plan: torch.Tensor) -> torch.Tensor:
    plan = plan.float()
    return plan / plan.sum(dim=0, keepdim=True).clamp_min(1.0e-8)


def _geometry(plan: torch.Tensor, occupancy: torch.Tensor) -> Dict[str, float]:
    plan = _column_normalize(plan.cpu())
    occupancy = _column_normalize(occupancy.cpu())
    num_frames, num_tokens = plan.shape
    frame = torch.arange(num_frames, dtype=plan.dtype).unsqueeze(1)
    plan_bary = (plan * frame).sum(dim=0)
    ctc_bary = (occupancy * frame).sum(dim=0)
    plan_support = plan >= 0.1 * plan.max(dim=0, keepdim=True).values
    ctc_support = occupancy >= 0.1 * occupancy.max(dim=0, keepdim=True).values
    support_union = (plan_support | ctc_support).sum().clamp_min(1)
    support_intersection = (plan_support & ctc_support).sum()

    if num_frames > 1 and num_tokens > 1:
        t = torch.linspace(0, 1, num_frames).unsqueeze(1)
        u = torch.linspace(0, 1, num_tokens).unsqueeze(0)
        distance = (t - u).abs()
        plan_diag = float((plan * distance).sum() / max(num_tokens, 1))
        ctc_diag = float((occupancy * distance).sum() / max(num_tokens, 1))
    else:
        plan_diag = ctc_diag = 0.0

    return {
        "plan_ctc_w1_frames": float(
            (plan.cumsum(dim=0) - occupancy.cumsum(dim=0))
            .abs()
            .sum()
            / max(num_tokens, 1)
        ),
        "plan_ctc_barycenter_mae_frames": float(
            (plan_bary - ctc_bary).abs().mean()
        ),
        "plan_ctc_support_iou": float(support_intersection / support_union),
        "plan_diagonal_deviation": plan_diag,
        "ctc_diagonal_deviation": ctc_diag,
    }


def _ctc_only_geometry(occupancy: torch.Tensor) -> Dict[str, float]:
    occupancy = _column_normalize(occupancy.cpu())
    num_frames, num_tokens = occupancy.shape
    frame = torch.arange(num_frames, dtype=occupancy.dtype).unsqueeze(1)
    barycenter = (occupancy * frame).sum(dim=0)
    support = occupancy >= 0.1 * occupancy.max(dim=0, keepdim=True).values
    if num_frames > 1 and num_tokens > 1:
        t = torch.linspace(0, 1, num_frames).unsqueeze(1)
        u = torch.linspace(0, 1, num_tokens).unsqueeze(0)
        diagonal = float(
            (occupancy * (t - u).abs()).sum() / max(num_tokens, 1)
        )
    else:
        diagonal = 0.0
    return {
        "plan_ctc_w1_frames": float("nan"),
        "plan_ctc_barycenter_mae_frames": float("nan"),
        "plan_ctc_support_iou": float("nan"),
        "plan_diagonal_deviation": float("nan"),
        "ctc_diagonal_deviation": diagonal,
        "ctc_barycenter_mean_frames": float(barycenter.mean()),
        "ctc_support_width_frames": float(support.sum(dim=0).float().mean()),
    }


def _build_model(
    checkpoint_path: Path,
    lang_dir: Path,
    device: torch.device,
    model_family: str = "auto",
    checkpoint_state: str = "model",
) -> Tuple[nn.Module, AttributeDict, str]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    ignored = {
        "model",
        "model_avg",
        "optimizer",
        "scheduler",
        "grad_scaler",
        "sampler",
    }
    params = AttributeDict(
        {key: value for key, value in checkpoint.items() if key not in ignored}
    )
    lexicon = Lexicon(lang_dir)
    num_classes = max(lexicon.tokens) + 1
    if model_family == "auto":
        architecture = str(getattr(params, "architecture", ""))
        model_family = "tdnn" if architecture.startswith("tdnn") else "vfta"
    state = checkpoint.get(checkpoint_state)
    if state is None:
        raise ValueError(
            f"{checkpoint_path} does not contain checkpoint state "
            f"{checkpoint_state!r}"
        )

    if model_family == "tdnn":
        model = TdnnLabelPriorCTC(
            num_features=int(getattr(params, "feature_dim", 80)),
            num_classes=num_classes,
            subsampling_factor=int(getattr(params, "subsampling_factor", 2)),
            d_model=int(getattr(params, "encoder_dim", 640)),
            num_decoder_layers=0,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return model, params, model_family

    conformer = Conformer(
        num_features=int(getattr(params, "feature_dim", 80)),
        nhead=int(getattr(params, "nhead", 8)),
        d_model=int(getattr(params, "encoder_dim", 512)),
        num_classes=num_classes,
        subsampling_factor=int(getattr(params, "subsampling_factor", 4)),
        dim_feedforward=int(getattr(params, "dim_feedforward", 2048)),
        num_encoder_layers=int(getattr(params, "num_encoder_layers", 12)),
        num_decoder_layers=int(getattr(params, "num_decoder_layers", 0)),
    )
    ctc_head = nn.Linear(int(params.encoder_dim), num_classes - 1)
    blank_gate = BlankGateHeadV2(
        d_model=int(params.encoder_dim),
        vocab_size=num_classes,
        d_attn=int(getattr(params, "label_embed_dim", 256)),
        init_blank_prob=float(getattr(params, "init_blank_prob", 0.1)),
    )
    blank_prior = BlankPriorHeadV2(
        d_model=int(params.encoder_dim),
        init_blank_prob=float(getattr(params, "init_blank_prob", 0.1)),
    )
    model = ConformerVIV2(conformer, ctc_head, blank_gate, blank_prior)
    if any(key.startswith("structural_metric.") for key in state):
        if str(getattr(params, "frame_metric", "learned-psd")) == "learned-diag-psd":
            model.structural_metric = DiagonalResidualPSDFrameMetric(
                feature_dim=int(getattr(params, "feature_dim", 80)),
                max_log_scale=float(getattr(params, "metric_max_log_scale", 0.5)),
            )
        else:
            model.structural_metric = LearnablePSDFrameMetric(
                feature_dim=int(getattr(params, "feature_dim", 80))
            )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, params, model_family


def _make_loader(cuts: CutSet, max_duration: float, num_workers: int) -> DataLoader:
    dataset = K2SpeechRecognitionDataset(
        input_strategy=PrecomputedFeatures(),
        return_cuts=True,
    )
    sampler = SimpleCutSampler(cuts, max_duration=max_duration, shuffle=False)
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _reconstruct_plan(
    log_p_nonblank: torch.Tensor,
    alpha: torch.Tensor,
    labels: torch.Tensor,
    raw_features: torch.Tensor,
    raw_num_frames: int,
    params: AttributeDict,
    structural_metric: nn.Module | None = None,
) -> torch.Tensor:
    common = dict(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        bpe_lengths=None,
        column_marginal_type=str(getattr(params, "col_marginal_type", "acoustic")),
        alpha_smooth_mix=float(getattr(params, "alpha_smooth_mix", 0.1)),
        bpe_col_floor=float(getattr(params, "bpe_col_floor", 0.05)),
        token_prior_sigma=float(getattr(params, "ot_token_prior_sigma", 0.15)),
        token_prior_score_temp=float(
            getattr(params, "ot_token_prior_score_temp", 1.0)
        ),
        token_prior_floor=float(getattr(params, "ot_token_prior_floor", 0.05)),
        eps=float(getattr(params, "ot_eps", 0.3)),
        iters=int(getattr(params, "ot_iters", 30)),
        beta_pos=float(getattr(params, "ot_beta_pos", 1.0)),
        return_plan=True,
    )
    if common["column_marginal_type"] == "bpe":
        raise ValueError("A phone checkpoint cannot use a BPE column marginal")

    lambda_gw = float(getattr(params, "lambda_gw", 0.0))
    if lambda_gw > 0:
        subsampling = int(getattr(params, "subsampling_factor", 4))
        indices = (
            torch.arange(log_p_nonblank.size(0), device=raw_features.device)
            * subsampling
            + subsampling // 2
        ).clamp(max=max(raw_num_frames - 1, 0))
        acoustic = raw_features.index_select(0, indices)
        _, plan = vi_fgw_loss_v2(
            **common,
            acoustic_features=acoustic,
            lambda_gw=lambda_gw,
            n_outer=int(getattr(params, "gw_n_outer", 3)),
            frame_metric=structural_metric,
            metric_rho=float(getattr(params, "metric_rho", 1.0)),
            metric_normalization=str(
                getattr(params, "metric_normalization", "none")
            ),
            metric_grad_scale=0.0,
            metric_moment_reg_weight=0.0,
            metric_identity_reg_weight=0.0,
        )
    else:
        _, plan = vi_ot_loss_v2(**common)
    if plan is None:
        raise RuntimeError("Failed to reconstruct OT/FGW plan")
    return plan


def _audit_split(
    cuts: Iterable,
    compiler: WordPhoneCtcTrainingGraphCompiler,
) -> Dict[str, object]:
    counts = {
        "utterances": 0,
        "target_words": 0,
        "gold_words": 0,
        "matched_words": 0,
        "target_phones": 0,
        "gold_nonsilence_phones": 0,
        "matched_phones": 0,
        "oov_words": 0,
    }
    oov: Dict[str, int] = {}
    for cut in cuts:
        transcript = compiler.expand_text(cut.supervisions[0].text)
        gold_words, _, _ = _gold_words(compiler, cut)
        gold_phones, _, _ = _gold_phones(cut)
        predicted_phones = [normalize_phone(p) for p in transcript.phones]
        word_pairs = _exact_match_pairs(gold_words, transcript.words)
        phone_pairs = _exact_match_pairs(gold_phones, predicted_phones)
        counts["utterances"] += 1
        counts["target_words"] += len(transcript.words)
        counts["gold_words"] += len(gold_words)
        counts["matched_words"] += len(word_pairs)
        counts["target_phones"] += len(predicted_phones)
        counts["gold_nonsilence_phones"] += len(gold_phones)
        counts["matched_phones"] += len(phone_pairs)
        counts["oov_words"] += len(transcript.oov_words)
        for word in transcript.oov_words:
            oov[word] = oov.get(word, 0) + 1
    return {
        **counts,
        "word_target_match_coverage": counts["matched_words"]
        / max(counts["target_words"], 1),
        "word_gold_match_coverage": counts["matched_words"]
        / max(counts["gold_words"], 1),
        "phone_target_match_coverage": counts["matched_phones"]
        / max(counts["target_phones"], 1),
        "phone_gold_match_coverage": counts["matched_phones"]
        / max(counts["gold_nonsilence_phones"], 1),
        "oov_rate": counts["oov_words"] / max(counts["target_words"], 1),
        "oov_histogram": dict(
            sorted(oov.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


@torch.inference_mode()
def _evaluate_split(
    cuts: CutSet,
    compiler: WordPhoneCtcTrainingGraphCompiler,
    model: nn.Module,
    params: AttributeDict,
    model_family: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    all_phone_duration: List[float] = []
    all_word_duration: List[float] = []
    all_gold_phone_duration: List[float] = []
    all_gold_word_duration: List[float] = []

    for batch in _make_loader(cuts, args.max_duration, args.num_workers):
        features = batch["inputs"].to(device)
        supervisions = batch["supervisions"]
        texts = list(supervisions["text"])
        sequence_idx = [int(i) for i in supervisions["sequence_idx"].tolist()]
        transcripts = [compiler.expand_text(text) for text in texts]

        batch_size = features.size(0)
        input_transcripts: List[PhoneTranscript] = [transcripts[0]] * batch_size
        for supervision_index, input_index in enumerate(sequence_idx):
            input_transcripts[input_index] = transcripts[supervision_index]
        target_lengths = torch.tensor(
            [len(t.phone_ids) for t in input_transcripts],
            dtype=torch.long,
            device=device,
        )
        targets = torch.zeros(
            (batch_size, int(target_lengths.max().item())),
            dtype=torch.long,
            device=device,
        )
        for i, transcript in enumerate(input_transcripts):
            targets[i, : len(transcript.phone_ids)] = torch.tensor(
                transcript.phone_ids, dtype=torch.long, device=device
            )

        if model_family == "vfta":
            (
                _,
                _,
                output_lens,
                log_p_nonblank,
                alpha_prior,
                alpha_post,
            ) = model(
                features,
                supervisions,
                targets=targets,
                target_lengths=target_lengths,
                warmup=1.0,
            )
            if args.prior_logit_bias != 0.0:
                alpha_prior = alpha_prior.float().clamp(1.0e-5, 1.0 - 1.0e-5)
                alpha_prior = torch.sigmoid(
                    torch.logit(alpha_prior) + args.prior_logit_bias
                )
            alpha = alpha_prior if args.gate == "prior" else alpha_post
            log_probs = build_gated_log_probs_v2(log_p_nonblank, alpha)
        else:
            base_log_probs, _, padding_mask = model(
                features, supervisions, warmup=1.0
            )
            output_lens = (~padding_mask).sum(dim=1)
            decode_alpha = (
                float(args.decode_label_prior_alpha)
                if args.decode_label_prior_alpha is not None
                else float(getattr(params, "label_prior_alpha", 0.0))
            )
            log_probs = model.prior_adjusted_log_probs(
                base_log_probs,
                alpha=decode_alpha,
                floor=float(
                    getattr(params, "label_prior_floor", math.exp(-12.0))
                ),
                enabled=True,
            )
            log_p_nonblank = alpha = None

        cuts_in_batch = supervisions["cut"]
        for supervision_index, cut in enumerate(cuts_in_batch):
            input_index = sequence_idx[supervision_index]
            transcript = transcripts[supervision_index]
            output_len = int(output_lens[input_index].item())
            lp = log_probs[input_index, :output_len]
            labels = transcript.phone_ids
            spans = _ctc_viterbi_token_frames(lp, labels)
            subsampling = int(getattr(params, "subsampling_factor", 4))
            phone_starts, phone_ends = _frames_to_intervals(
                spans,
                float(cut.duration),
                subsampling,
                model_family=model_family,
            )
            word_starts, word_ends = _word_intervals(
                transcript, phone_starts, phone_ends
            )

            gold_words, gold_word_starts, gold_word_ends = _gold_words(
                compiler, cut
            )
            gold_phones, gold_phone_starts, gold_phone_ends = _gold_phones(cut)
            predicted_phones = [normalize_phone(p) for p in transcript.phones]
            word_pairs = _exact_match_pairs(gold_words, transcript.words)
            phone_pairs = _exact_match_pairs(gold_phones, predicted_phones)
            phone_errors = _paired_errors(
                phone_pairs,
                phone_starts,
                phone_ends,
                gold_phone_starts,
                gold_phone_ends,
            )
            word_errors = _paired_errors(
                word_pairs,
                word_starts,
                word_ends,
                gold_word_starts,
                gold_word_ends,
            )

            all_phone_duration.extend(
                1000.0 * (end - start)
                for start, end in zip(phone_starts, phone_ends)
            )
            all_word_duration.extend(
                1000.0 * (end - start)
                for start, end in zip(word_starts, word_ends)
            )
            all_gold_phone_duration.extend(
                1000.0 * (end - start)
                for start, end in zip(gold_phone_starts, gold_phone_ends)
            )
            all_gold_word_duration.extend(
                1000.0 * (end - start)
                for start, end in zip(gold_word_starts, gold_word_ends)
            )

            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            occupancy = ctc_token_occupancy_batched(
                log_probs=lp.unsqueeze(0),
                labels=labels_t.unsqueeze(0),
                frame_lens=torch.tensor([output_len], device=device),
                label_lens=torch.tensor([len(labels)], device=device),
                blank_id=0,
            )[0]
            if model_family == "vfta":
                plan = _reconstruct_plan(
                    log_p_nonblank=log_p_nonblank[input_index, :output_len],
                    alpha=alpha[input_index, :output_len],
                    labels=labels_t,
                    raw_features=features[input_index],
                    raw_num_frames=int(
                        supervisions["num_frames"][supervision_index]
                    ),
                    params=params,
                    structural_metric=getattr(model, "structural_metric", None),
                )
                geometry = {
                    **_geometry(plan, occupancy),
                    "ctc_barycenter_mean_frames": float("nan"),
                    "ctc_support_width_frames": float("nan"),
                }
            else:
                geometry = _ctc_only_geometry(occupancy)

            rows.append(
                {
                    "cut_id": cut.id,
                    "oov_words": transcript.oov_words,
                    "phone_target_count": len(predicted_phones),
                    "phone_gold_count": len(gold_phones),
                    "phone_match_count": len(phone_pairs),
                    "phone_match_coverage_target": len(phone_pairs)
                    / max(len(predicted_phones), 1),
                    "phone_match_coverage_gold": len(phone_pairs)
                    / max(len(gold_phones), 1),
                    "PBE_ms": _safe_mean(phone_errors["boundary"]),
                    "phone_onset_mae_ms": _safe_mean(phone_errors["onset"]),
                    "phone_offset_mae_ms": _safe_mean(phone_errors["offset"]),
                    "word_target_count": len(transcript.words),
                    "word_gold_count": len(gold_words),
                    "word_match_count": len(word_pairs),
                    "word_match_coverage_target": len(word_pairs)
                    / max(len(transcript.words), 1),
                    "word_match_coverage_gold": len(word_pairs)
                    / max(len(gold_words), 1),
                    "WBE_ms": _safe_mean(word_errors["boundary"]),
                    "word_onset_mae_ms": _safe_mean(word_errors["onset"]),
                    "word_offset_mae_ms": _safe_mean(word_errors["offset"]),
                    **geometry,
                }
            )

    def macro(key: str) -> float:
        values = [
            float(row[key])
            for row in rows
            if not math.isnan(float(row[key]))
        ]
        return _safe_mean(values)

    summary = {
        "checkpoint": str(args.checkpoint),
        "model_family": model_family,
        "checkpoint_state": args.checkpoint_state,
        "gate": args.gate,
        "prior_logit_bias": args.prior_logit_bias,
        "decode_label_prior_alpha": (
            args.decode_label_prior_alpha
            if args.decode_label_prior_alpha is not None
            else float(getattr(params, "label_prior_alpha", 0.0))
        ),
        "num_utterances": len(rows),
        "PBE_ms_utterance_macro": macro("PBE_ms"),
        "WBE_ms_utterance_macro": macro("WBE_ms"),
        "phone_onset_mae_ms_utterance_macro": macro("phone_onset_mae_ms"),
        "phone_offset_mae_ms_utterance_macro": macro("phone_offset_mae_ms"),
        "word_onset_mae_ms_utterance_macro": macro("word_onset_mae_ms"),
        "word_offset_mae_ms_utterance_macro": macro("word_offset_mae_ms"),
        "PDUR_ms": _safe_mean(all_phone_duration),
        "WDUR_ms": _safe_mean(all_word_duration),
        "gold_PDUR_ms": _safe_mean(all_gold_phone_duration),
        "gold_WDUR_ms": _safe_mean(all_gold_word_duration),
        "phone_match_coverage_target": sum(
            int(row["phone_match_count"]) for row in rows
        )
        / max(sum(int(row["phone_target_count"]) for row in rows), 1),
        "phone_match_coverage_gold": sum(
            int(row["phone_match_count"]) for row in rows
        )
        / max(sum(int(row["phone_gold_count"]) for row in rows), 1),
        "word_match_coverage_target": sum(
            int(row["word_match_count"]) for row in rows
        )
        / max(sum(int(row["word_target_count"]) for row in rows), 1),
        "word_match_coverage_gold": sum(
            int(row["word_match_count"]) for row in rows
        )
        / max(sum(int(row["word_gold_count"]) for row in rows), 1),
        "plan_ctc_w1_frames": macro("plan_ctc_w1_frames"),
        "plan_ctc_barycenter_mae_frames": macro(
            "plan_ctc_barycenter_mae_frames"
        ),
        "plan_ctc_support_iou": macro("plan_ctc_support_iou"),
        "plan_diagonal_deviation": macro("plan_diagonal_deviation"),
        "ctc_diagonal_deviation": macro("ctc_diagonal_deviation"),
        "ctc_barycenter_mean_frames": macro("ctc_barycenter_mean_frames"),
        "ctc_support_width_frames": macro("ctc_support_width_frames"),
        "aggregation": (
            "PBE/WBE and onset/offset are utterance-macro means. Duration is "
            "pooled over predicted or gold items. PBE uses only exact phone "
            "matches after stress stripping; coverage is mandatory context."
        ),
    }
    return summary, rows


def main() -> None:
    args = get_parser().parse_args()
    if not args.audit_only and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --audit-only is set")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lexicon = Lexicon(args.lang_dir)
    compiler = WordPhoneCtcTrainingGraphCompiler(
        lang_dir=args.lang_dir,
        lexicon=lexicon,
        device=torch.device("cpu"),
    )

    split_cuts: Dict[str, CutSet] = {}
    for split in args.splits:
        manifest_dir = args.timit_manifest_dir.resolve()
        path = manifest_dir / f"timit_cuts_{split}.jsonl.gz"
        cuts = CutSet.from_file(path)
        # TIMIT manifests store paths relative to egs/timit/ASR.  Evaluation is
        # intentionally launched from the LibriSpeech recipe, so resolve them
        # here instead of relying on the process working directory.
        timit_recipe_dir = manifest_dir.parents[1]
        cuts = cuts.with_features_path_prefix(
            timit_recipe_dir
        ).with_recording_path_prefix(timit_recipe_dir)
        if not args.include_sa:
            cuts = CutSet.from_cuts(cut for cut in cuts if "-SA" not in cut.id)
            expected = {"DEV": 400, "TEST": 192}[split]
            if len(cuts) != expected:
                raise ValueError(
                    f"{split}: expected {expected} no-SA cuts, found {len(cuts)}"
                )
        if args.max_cuts > 0:
            cuts = cuts.subset(first=args.max_cuts)
        split_cuts[split] = cuts

    audit = {
        split: _audit_split(cuts, compiler)
        for split, cuts in split_cuts.items()
    }
    with open(args.output_dir / "protocol-audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, params, model_family = _build_model(
        args.checkpoint,
        args.lang_dir,
        device,
        model_family=args.model_family,
        checkpoint_state=args.checkpoint_state,
    )
    summaries: Dict[str, object] = {
        "configuration": {
            "training_corpus": "LibriSpeech",
            "timit_finetuning": False,
            "include_sa": args.include_sa,
            "target_source": "orthographic text expanded by the Libri lexicon",
        },
        "protocol_audit": audit,
    }
    for split, cuts in split_cuts.items():
        summary, rows = _evaluate_split(
            cuts=cuts,
            compiler=compiler,
            model=model,
            params=params,
            model_family=model_family,
            args=args,
            device=device,
        )
        summaries[split] = summary
        with open(args.output_dir / f"{split.lower()}-details.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
