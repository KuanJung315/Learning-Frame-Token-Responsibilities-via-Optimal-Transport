#!/usr/bin/env python3
"""Compare two LibriSpeech phone-VFTA checkpoints against MFA boundaries.

The phone transcript is deterministically expanded from the orthographic
LibriSpeech transcript using exactly the same lexicon/compiler as training.
MFA is an automatic pseudo-reference, not manually annotated ground truth.

In addition to external word-boundary metrics, this evaluator reconstructs
each checkpoint's OT/FGW plan and compares it with differentiable, blank-aware
CTC token occupancy.  This separates "the plan moved" from "CTC geometry
followed the plan."
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent
FGW_DIR = SCRIPT_DIR / "vfta_fgw"
for path in (SCRIPT_DIR, ASR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(FGW_DIR) not in sys.path:
    # Append so vfta_fgw/train.py cannot shadow this recipe's train.py.
    sys.path.append(str(FGW_DIR))

from asr_datamodule import LibriSpeechAsrDataModule
from blank_gate_v2 import BlankGateHeadV2, BlankPriorHeadV2
from conformer import Conformer
from ctc_plan_consistency import ctc_token_occupancy_batched
from evaluate_alignment_metrics import _load_eval_dataloader
from evaluate_mfa_alignment import (
    _reference_word_spans,
    _word_rows,
    ctc_viterbi_align,
    librispeech_utterance_id,
    load_mfa_manifest,
    summarize,
    token_spans_to_words,
    write_csv,
    write_summary_markdown,
)
from ot_fgw import (
    DiagonalResidualPSDFrameMetric,
    LearnablePSDFrameMetric,
    vi_fgw_loss_v2,
)
from ot_prior_v2 import vi_ot_loss_v2
from varctc_v2_utils import build_gated_log_probs_v2
from varctc_v2_utils import encoder_lens_from_mask
from word_phone_graph_compiler import (
    PhoneTranscript,
    WordPhoneCtcTrainingGraphCompiler,
)

from icefall.lexicon import Lexicon
from icefall.utils import AttributeDict, setup_logger


class PhoneVFTAForAlignment(nn.Module):
    """Checkpoint-compatible VFTA module without importing the train program."""

    def __init__(
        self,
        encoder: Conformer,
        ctc_head: nn.Linear,
        blank_gate: BlankGateHeadV2,
        blank_prior: BlankPriorHeadV2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.ctc_head = ctc_head
        self.blank_gate = blank_gate
        self.blank_prior = blank_prior

    def forward(
        self,
        x: torch.Tensor,
        supervisions: Mapping[str, Any],
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        warmup: float = 1.0,
    ):
        memory, memory_key_padding_mask = self.encoder.run_encoder(
            x, supervisions, warmup=warmup
        )
        hidden = memory.permute(1, 0, 2)
        output_lens = encoder_lens_from_mask(
            memory_key_padding_mask,
            batch_size=hidden.size(0),
            max_len=hidden.size(1),
            device=hidden.device,
        )
        log_p_nonblank = nn.functional.log_softmax(
            self.ctc_head(hidden), dim=-1
        )
        alpha_prior = self.blank_prior(hidden, output_lens)
        alpha_post = self.blank_gate(
            hidden, targets, target_lengths, output_lens
        )
        return (
            memory,
            memory_key_padding_mask,
            output_lens,
            log_p_nonblank,
            alpha_prior,
            alpha_post,
        )


def _build_model(
    checkpoint_path: Path,
    lang_dir: Path,
    device: torch.device,
) -> Tuple[PhoneVFTAForAlignment, AttributeDict]:
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
    encoder_dim = int(getattr(params, "encoder_dim", 512))
    encoder = Conformer(
        num_features=int(getattr(params, "feature_dim", 80)),
        nhead=int(getattr(params, "nhead", 8)),
        d_model=encoder_dim,
        num_classes=num_classes,
        subsampling_factor=int(getattr(params, "subsampling_factor", 4)),
        dim_feedforward=int(getattr(params, "dim_feedforward", 2048)),
        num_encoder_layers=int(getattr(params, "num_encoder_layers", 12)),
        num_decoder_layers=int(getattr(params, "num_decoder_layers", 0)),
    )
    model = PhoneVFTAForAlignment(
        encoder=encoder,
        ctc_head=nn.Linear(encoder_dim, num_classes - 1),
        blank_gate=BlankGateHeadV2(
            d_model=encoder_dim,
            vocab_size=num_classes,
            d_attn=int(getattr(params, "label_embed_dim", 256)),
            init_blank_prob=float(getattr(params, "init_blank_prob", 0.1)),
        ),
        blank_prior=BlankPriorHeadV2(
            d_model=encoder_dim,
            init_blank_prob=float(getattr(params, "init_blank_prob", 0.1)),
        ),
    )
    state = checkpoint["model"]
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
    return model, params


def _column_normalize(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.detach().float().cpu()
    return matrix / matrix.sum(dim=0, keepdim=True).clamp_min(1.0e-8)


def _geometry(
    plan: torch.Tensor, occupancy: torch.Tensor
) -> Dict[str, float]:
    plan = _column_normalize(plan)
    occupancy = _column_normalize(occupancy)
    num_frames, num_tokens = plan.shape
    frame = torch.arange(num_frames, dtype=plan.dtype).unsqueeze(1)
    plan_barycenter = (plan * frame).sum(dim=0)
    ctc_barycenter = (occupancy * frame).sum(dim=0)
    plan_support = plan >= 0.1 * plan.max(dim=0, keepdim=True).values
    ctc_support = occupancy >= 0.1 * occupancy.max(dim=0, keepdim=True).values
    support_union = (plan_support | ctc_support).sum().clamp_min(1)
    support_intersection = (plan_support & ctc_support).sum()

    if num_frames > 1 and num_tokens > 1:
        time_position = torch.linspace(0, 1, num_frames).unsqueeze(1)
        token_position = torch.linspace(0, 1, num_tokens).unsqueeze(0)
        diagonal_distance = (time_position - token_position).abs()
        plan_diagonal = float(
            (plan * diagonal_distance).sum() / max(num_tokens, 1)
        )
        ctc_diagonal = float(
            (occupancy * diagonal_distance).sum() / max(num_tokens, 1)
        )
    else:
        plan_diagonal = ctc_diagonal = 0.0
    return {
        "plan_ctc_w1_frames": float(
            (plan.cumsum(dim=0) - occupancy.cumsum(dim=0))
            .abs()
            .sum()
            / max(num_tokens, 1)
        ),
        "plan_ctc_barycenter_mae_frames": float(
            (plan_barycenter - ctc_barycenter).abs().mean()
        ),
        "plan_ctc_support_iou": float(
            support_intersection / support_union
        ),
        "plan_diagonal_deviation": plan_diagonal,
        "ctc_diagonal_deviation": ctc_diagonal,
    }


def _encoder_aligned_raw_features(
    raw_features: torch.Tensor,
    raw_num_frames: int,
    output_len: int,
    subsampling_factor: int,
) -> torch.Tensor:
    """Select log-Mel frames aligned with the encoder output time axis."""
    if raw_num_frames <= 0 or output_len <= 0:
        raise ValueError("raw_num_frames and output_len must be positive")
    if subsampling_factor <= 0:
        raise ValueError("subsampling_factor must be positive")
    indices = (
        torch.arange(output_len, device=raw_features.device)
        * subsampling_factor
        + subsampling_factor // 2
    ).clamp(max=raw_num_frames - 1)
    return raw_features.index_select(0, indices)


def _reconstruct_plan(
    log_p_nonblank: torch.Tensor,
    alpha: torch.Tensor,
    labels: torch.Tensor,
    raw_features: torch.Tensor,
    raw_num_frames: int,
    params: AttributeDict,
    structural_metric: nn.Module | None = None,
    metric_rho: float | None = None,
) -> torch.Tensor:
    common = dict(
        log_p_nonblank=log_p_nonblank,
        alpha=alpha,
        labels=labels,
        bpe_lengths=None,
        column_marginal_type=str(
            getattr(params, "col_marginal_type", "acoustic")
        ),
        alpha_smooth_mix=float(getattr(params, "alpha_smooth_mix", 0.1)),
        bpe_col_floor=float(getattr(params, "bpe_col_floor", 0.05)),
        token_prior_sigma=float(
            getattr(params, "ot_token_prior_sigma", 0.15)
        ),
        token_prior_score_temp=float(
            getattr(params, "ot_token_prior_score_temp", 1.0)
        ),
        token_prior_floor=float(
            getattr(params, "ot_token_prior_floor", 0.05)
        ),
        eps=float(getattr(params, "ot_eps", 0.3)),
        iters=int(getattr(params, "ot_iters", 30)),
        beta_pos=float(getattr(params, "ot_beta_pos", 1.0)),
        return_plan=True,
    )
    if common["column_marginal_type"] == "bpe":
        raise ValueError("A phone checkpoint cannot use a BPE column marginal")

    lambda_gw = float(getattr(params, "lambda_gw", 0.0))
    if lambda_gw > 0.0:
        subsampling = int(getattr(params, "subsampling_factor", 4))
        acoustic_features = _encoder_aligned_raw_features(
            raw_features=raw_features,
            raw_num_frames=raw_num_frames,
            output_len=log_p_nonblank.size(0),
            subsampling_factor=subsampling,
        )
        _, plan = vi_fgw_loss_v2(
            **common,
            acoustic_features=acoustic_features,
            lambda_gw=lambda_gw,
            n_outer=int(getattr(params, "gw_n_outer", 3)),
            frame_metric=structural_metric,
            metric_rho=(
                float(getattr(params, "metric_rho", 1.0))
                if metric_rho is None
                else float(metric_rho)
            ),
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
        raise RuntimeError("Failed to reconstruct the OT/FGW plan")
    return plan


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-name", type=str, default="plan_w1")
    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=Path("data/lang_phone_nostress"),
        help="The exact deterministic phone language directory used in training.",
    )
    parser.add_argument(
        "--mfa-dir", type=Path, default=Path("data/librispeech_mfa")
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["dev-clean", "dev-other"]
    )
    parser.add_argument("--max-cuts-per-dataset", type=int, default=0)
    parser.add_argument(
        "--gate",
        choices=["prior", "posterior"],
        default="prior",
        help=(
            "Prior is the primary transcript-independent gate. Posterior is a "
            "separate transcript-conditioned diagnostic."
        ),
    )
    parser.add_argument("--baseline-prior-logit-bias", type=float, default=0.0)
    parser.add_argument("--candidate-prior-logit-bias", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/mfa_phone_alignment_eval"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame-shift-ms", type=float, default=10.0)
    parser.add_argument("--subsampling-factor", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    LibriSpeechAsrDataModule.add_arguments(parser)
    return parser


def _apply_prior_bias(alpha: torch.Tensor, bias: float) -> torch.Tensor:
    if bias == 0.0:
        return alpha
    alpha = alpha.float().clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.sigmoid(torch.logit(alpha) + bias)


def _input_order_transcripts(
    transcripts: Sequence[PhoneTranscript],
    sequence_idx: Sequence[int],
    batch_size: int,
) -> List[PhoneTranscript]:
    if len(transcripts) != len(sequence_idx):
        raise ValueError("A sequence_idx entry is required for every supervision")
    if not transcripts:
        raise ValueError("An empty supervision batch cannot be aligned")
    ordered: List[PhoneTranscript | None] = [None] * batch_size
    for supervision_index, input_index in enumerate(sequence_idx):
        if not 0 <= input_index < batch_size:
            raise ValueError(f"Invalid sequence_idx: {input_index}")
        if ordered[input_index] is not None:
            raise ValueError("Multiple supervisions per cut are not supported")
        ordered[input_index] = transcripts[supervision_index]
    if any(transcript is None for transcript in ordered):
        raise ValueError("Every input cut must have exactly one supervision")
    return [transcript for transcript in ordered if transcript is not None]


def _padded_phone_targets(
    transcripts: Sequence[PhoneTranscript], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(
        [len(transcript.phone_ids) for transcript in transcripts],
        dtype=torch.long,
        device=device,
    )
    if lengths.numel() == 0 or int(lengths.min().item()) <= 0:
        raise ValueError("Every transcript must contain at least one phone")
    targets = torch.zeros(
        (len(transcripts), int(lengths.max().item())),
        dtype=torch.long,
        device=device,
    )
    for index, transcript in enumerate(transcripts):
        targets[index, : len(transcript.phone_ids)] = torch.tensor(
            transcript.phone_ids, dtype=torch.long, device=device
        )
    return targets, lengths


@torch.inference_mode()
def _model_outputs(
    model,
    features: torch.Tensor,
    supervisions: Mapping[str, Any],
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    gate: str,
    prior_logit_bias: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    alpha_prior = _apply_prior_bias(alpha_prior, prior_logit_bias)
    alpha = alpha_prior if gate == "prior" else alpha_post
    log_probs = build_gated_log_probs_v2(log_p_nonblank, alpha)
    return log_probs, output_lens, log_p_nonblank, alpha


def _normalized_reference_words(
    compiler: WordPhoneCtcTrainingGraphCompiler,
    reference: Sequence[Tuple[str, float, float]],
) -> List[str] | None:
    normalized: List[str] = []
    for word, _, _ in reference:
        words = compiler.normalize_words(word)
        if len(words) != 1:
            return None
        normalized.append(words[0])
    return normalized


def _describe(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "std": float(array.std()),
    }


def summarize_geometry(
    rows: Sequence[Mapping[str, Any]], candidate_name: str
) -> Dict[str, Any]:
    metric_names = (
        "plan_ctc_w1_frames",
        "plan_ctc_barycenter_mae_frames",
        "plan_ctc_support_iou",
        "plan_diagonal_deviation",
        "ctc_diagonal_deviation",
    )
    datasets = sorted({str(row["dataset"]) for row in rows})
    models = ("baseline", candidate_name)
    output: Dict[str, Any] = {}
    for dataset in datasets + ["combined"]:
        selected = [
            row
            for row in rows
            if dataset == "combined" or str(row["dataset"]) == dataset
        ]
        model_summary: Dict[str, Any] = {}
        for model in models:
            model_rows = [row for row in selected if row["model"] == model]
            if model_rows:
                model_summary[model] = {
                    "num_utterances": len(model_rows),
                    "metrics": {
                        metric: _describe([float(row[metric]) for row in model_rows])
                        for metric in metric_names
                    },
                }

        pairs: Dict[str, Dict[str, float]] = {}
        by_cut: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in selected:
            by_cut[str(row["cut_id"])][str(row["model"])] = row
        for metric in metric_names:
            deltas = [
                float(model_rows[candidate_name][metric])
                - float(model_rows["baseline"][metric])
                for model_rows in by_cut.values()
                if "baseline" in model_rows and candidate_name in model_rows
            ]
            if deltas:
                pairs[metric] = {
                    "candidate_minus_baseline": float(np.mean(deltas)),
                    "num_paired_utterances": len(deltas),
                }
        output[dataset] = {"models": model_summary, "paired": pairs}
    return output


def write_geometry_csv(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    if not rows:
        raise ValueError("No utterance-level geometry rows were produced")
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    lexicon = Lexicon(args.lang_dir)
    compiler = WordPhoneCtcTrainingGraphCompiler(
        lang_dir=args.lang_dir, lexicon=lexicon, device=device
    )
    baseline, baseline_params = _build_model(
        args.baseline_checkpoint, args.lang_dir, device
    )
    candidate, candidate_params = _build_model(
        args.candidate_checkpoint, args.lang_dir, device
    )
    for name, params in (
        ("baseline", baseline_params),
        (args.candidate_name, candidate_params),
    ):
        factor = int(getattr(params, "subsampling_factor", 4))
        if factor != args.subsampling_factor:
            raise ValueError(
                f"{name} checkpoint subsampling={factor}, but evaluator "
                f"--subsampling-factor={args.subsampling_factor}"
            )

    seconds_per_frame = (
        args.frame_shift_ms * args.subsampling_factor / 1000.0
    )
    word_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    skipped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    model_specs = (
        (
            "baseline",
            baseline,
            baseline_params,
            args.baseline_prior_logit_bias,
        ),
        (
            args.candidate_name,
            candidate,
            candidate_params,
            args.candidate_prior_logit_bias,
        ),
    )

    for dataset in args.datasets:
        references = load_mfa_manifest(args.mfa_dir, dataset)
        args.dataset = dataset
        args.max_cuts = args.max_cuts_per_dataset
        dataloader = _load_eval_dataloader(args)
        processed = 0
        for batch_index, batch in enumerate(dataloader):
            features = batch["inputs"].to(device)
            supervisions = batch["supervisions"]
            texts: Sequence[str] = list(supervisions["text"])
            cuts = supervisions["cut"]
            sequence_idx = [
                int(index) for index in supervisions["sequence_idx"].tolist()
            ]
            transcripts = [compiler.expand_text(text) for text in texts]
            input_transcripts = _input_order_transcripts(
                transcripts, sequence_idx, features.size(0)
            )
            targets, target_lengths = _padded_phone_targets(
                input_transcripts, device
            )
            outputs = {
                model_name: (
                    *_model_outputs(
                        model=model,
                        features=features,
                        supervisions=supervisions,
                        targets=targets,
                        target_lengths=target_lengths,
                        gate=args.gate,
                        prior_logit_bias=prior_bias,
                    ),
                    params,
                    model,
                )
                for model_name, model, params, prior_bias in model_specs
            }

            for supervision_index, (text, cut, transcript) in enumerate(
                zip(texts, cuts, transcripts)
            ):
                input_index = sequence_idx[supervision_index]
                cut_id = librispeech_utterance_id(cut)
                record = references.get(cut_id)
                if record is None or not record.get("words"):
                    skipped[dataset].append(
                        {"cut_id": cut_id, "reason": "missing_mfa_words"}
                    )
                    continue
                reference = _reference_word_spans(record)
                normalized_reference = _normalized_reference_words(
                    compiler, reference
                )
                if normalized_reference != transcript.words:
                    skipped[dataset].append(
                        {"cut_id": cut_id, "reason": "transcript_mismatch"}
                    )
                    continue
                if transcript.oov_words:
                    skipped[dataset].append(
                        {
                            "cut_id": cut_id,
                            "reason": "oov:" + ",".join(transcript.oov_words),
                        }
                    )
                    continue

                try:
                    for model_name, output in outputs.items():
                        (
                            log_probs,
                            output_lens,
                            log_p_nonblank,
                            alpha,
                            params,
                            model,
                        ) = output
                        output_len = int(output_lens[input_index].item())
                        lp = log_probs[input_index, :output_len]
                        token_spans, path_score = ctc_viterbi_align(
                            lp, transcript.phone_ids, blank_id=0
                        )
                        predicted = token_spans_to_words(
                            token_spans=token_spans,
                            token_ranges=transcript.word_phone_spans,
                            seconds_per_frame=seconds_per_frame,
                            duration=float(record["duration"]),
                        )
                        word_rows.extend(
                            _word_rows(
                                dataset=dataset,
                                cut_id=cut_id,
                                model_name=model_name,
                                reference=reference,
                                predicted=predicted,
                                path_score=path_score,
                                num_frames=output_len,
                            )
                        )

                        labels = torch.tensor(
                            transcript.phone_ids, dtype=torch.long, device=device
                        )
                        plan = _reconstruct_plan(
                            log_p_nonblank=log_p_nonblank[
                                input_index, :output_len
                            ],
                            alpha=alpha[input_index, :output_len],
                            labels=labels,
                            raw_features=features[input_index],
                            raw_num_frames=int(
                                supervisions["num_frames"][supervision_index]
                            ),
                            params=params,
                            structural_metric=getattr(
                                model, "structural_metric", None
                            ),
                        )
                        occupancy = ctc_token_occupancy_batched(
                            log_probs=lp.unsqueeze(0),
                            labels=labels.unsqueeze(0),
                            frame_lens=torch.tensor(
                                [output_len], dtype=torch.long, device=device
                            ),
                            label_lens=torch.tensor(
                                [len(transcript.phone_ids)],
                                dtype=torch.long,
                                device=device,
                            ),
                            blank_id=0,
                        )[0]
                        geometry_rows.append(
                            {
                                "dataset": dataset,
                                "cut_id": cut_id,
                                "model": model_name,
                                "num_frames": output_len,
                                "num_phones": len(transcript.phone_ids),
                                **_geometry(plan, occupancy),
                            }
                        )
                except (RuntimeError, ValueError) as error:
                    # Remove an incomplete model pair so every reported
                    # candidate-minus-baseline comparison stays paired.
                    word_rows[:] = [
                        row
                        for row in word_rows
                        if not (
                            row["dataset"] == dataset and row["cut_id"] == cut_id
                        )
                    ]
                    geometry_rows[:] = [
                        row
                        for row in geometry_rows
                        if not (
                            row["dataset"] == dataset and row["cut_id"] == cut_id
                        )
                    ]
                    skipped[dataset].append(
                        {"cut_id": cut_id, "reason": str(error)}
                    )
                    continue
                processed += 1

            if (batch_index + 1) % 10 == 0:
                logging.info(
                    "%s: processed %d utterances (%d batches)",
                    dataset,
                    processed,
                    batch_index + 1,
                )
        logging.info(
            "%s complete: processed=%d skipped=%d",
            dataset,
            processed,
            len(skipped[dataset]),
        )
    return word_rows, geometry_rows, dict(skipped)


def main() -> None:
    args = get_parser().parse_args()
    if args.candidate_name == "baseline":
        raise ValueError("--candidate-name must differ from 'baseline'")
    if not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(args.baseline_checkpoint)
    if not args.candidate_checkpoint.is_file():
        raise FileNotFoundError(args.candidate_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(f"{args.output_dir}/log-mfa-phone-alignment")
    logging.info("Arguments: %s", vars(args))

    word_rows, geometry_rows, skipped = evaluate(args)
    summary = summarize(
        word_rows,
        candidate_name=args.candidate_name,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["geometry"] = summarize_geometry(
        geometry_rows, candidate_name=args.candidate_name
    )
    summary["reference"] = {
        "name": "LibriSpeech MFA alignments",
        "type": "automatic pseudo-reference",
        "mfa_dir": str(args.mfa_dir),
    }
    summary["configuration"] = {
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "lang_dir": str(args.lang_dir),
        "gate": args.gate,
        "baseline_prior_logit_bias": args.baseline_prior_logit_bias,
        "candidate_prior_logit_bias": args.candidate_prior_logit_bias,
        "seconds_per_output_frame": (
            args.frame_shift_ms * args.subsampling_factor / 1000.0
        ),
        "checkpoint_selection_rule": (
            "Select on LibriSpeech development MFA WBE plus CTC geometry; "
            "do not select on TIMIT TEST."
        ),
    }
    write_csv(word_rows, args.output_dir / "word_metrics.csv")
    write_geometry_csv(
        geometry_rows, args.output_dir / "utterance_geometry.csv"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "skipped.json").write_text(
        json.dumps(skipped, indent=2) + "\n", encoding="utf-8"
    )
    write_summary_markdown(summary, args.output_dir / "summary.md")
    logging.info("Wrote phone MFA evaluation to %s", args.output_dir)


if __name__ == "__main__":
    main()
