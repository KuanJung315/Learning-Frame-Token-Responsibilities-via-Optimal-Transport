#!/usr/bin/env python3
"""Evaluate transcript-constrained CTC word spans against LibriSpeech MFA.

This is an external, timestamp-based evaluation.  It complements the existing
posterior-geometry diagnostics, which do not use a time-aligned reference.
MFA is an automatic pseudo-reference and must not be described as manually
annotated ground truth in a paper.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ASR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adamer_ctc.model import AdaMERConformer
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from evaluate_alignment_metrics import _checkpoint_metadata, _load_eval_dataloader
from evaluate_vi_alignment_metrics import (
    _batch_log_probs,
    _build_baseline_model,
    _build_candidate_model,
    _load_state,
)
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.utils import setup_logger, str2bool
from label_prior_ctc.model import LabelPriorConformer
from train import get_params as get_baseline_params


TOLERANCES_MS = (20, 40, 80, 100, 200)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--baseline-exp-dir", type=Path, required=True)
    parser.add_argument("--baseline-epoch", type=int, default=40)
    parser.add_argument("--baseline-avg", type=int, default=10)
    parser.add_argument("--baseline-use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--baseline-num-decoder-layers", type=int, default=6)
    parser.add_argument("--baseline-ctc-nonblank-logit-bias", type=float, default=0.0)

    parser.add_argument("--candidate-exp-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", type=str, default="vfta")
    parser.add_argument(
        "--candidate-kind",
        choices=["vfta", "label_prior", "adamer", "baseline"],
        default="vfta",
        help="Checkpoint architecture and inference-time score transform.",
    )
    parser.add_argument("--candidate-epoch", type=int, default=40)
    parser.add_argument("--candidate-avg", type=int, default=10)
    parser.add_argument("--candidate-use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--candidate-num-decoder-layers", type=int, default=6)
    parser.add_argument("--candidate-prior-logit-bias", type=float, default=0.0)
    parser.add_argument(
        "--candidate-ctc-nonblank-logit-bias", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate-label-prior-alpha",
        type=float,
        default=0.3,
        help=(
            "For candidate-kind=label_prior, apply log y - alpha log P(k) "
            "during forced alignment, matching the paper's Viterbi protocol."
        ),
    )
    parser.add_argument(
        "--candidate-label-prior-floor", type=float, default=math.exp(-12.0)
    )
    parser.add_argument("--label-embed-dim", type=int, default=256)
    parser.add_argument("--init-blank-prob", type=float, default=0.35)

    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_bpe_500"))
    parser.add_argument(
        "--mfa-dir", type=Path, default=Path("data/librispeech_mfa")
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["dev-clean", "dev-other"]
    )
    parser.add_argument("--max-cuts-per-dataset", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/mfa_alignment_eval"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame-shift-ms", type=float, default=10.0)
    parser.add_argument("--subsampling-factor", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    LibriSpeechAsrDataModule.add_arguments(parser)
    return parser


def _checkpoint_path(exp_dir: Path, epoch: int) -> Path:
    path = exp_dir / f"epoch-{epoch}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _build_standard_candidate(
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> Conformer:
    model_classes = {
        "baseline": Conformer,
        "adamer": AdaMERConformer,
        "label_prior": LabelPriorConformer,
    }
    model_cls = model_classes[args.candidate_kind]
    saved = _checkpoint_metadata(
        _checkpoint_path(args.candidate_exp_dir, args.candidate_epoch)
    )
    params = get_baseline_params()
    params.update(saved)
    if not hasattr(params, "num_decoder_layers"):
        params.num_decoder_layers = args.candidate_num_decoder_layers
    model = model_cls(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )
    _load_state(
        model=model,
        exp_dir=args.candidate_exp_dir,
        epoch=args.candidate_epoch,
        avg=args.candidate_avg,
        use_averaged_model=args.candidate_use_averaged_model,
        device=device,
    )
    if isinstance(model, LabelPriorConformer):
        if not bool(model.label_prior_ready.item()):
            raise RuntimeError("Label-prior checkpoint does not contain a ready prior")
        model.apply_prior_in_forward = args.candidate_label_prior_alpha > 0.0
        model.decode_alpha = args.candidate_label_prior_alpha
        model.decode_floor = args.candidate_label_prior_floor
    return model


def _build_evaluation_candidate(
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    if args.candidate_kind == "vfta":
        return _build_candidate_model(args, num_classes, device)
    return _build_standard_candidate(args, num_classes, device)


def ctc_viterbi_align(
    log_probs: torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
    blank_id: int = 0,
) -> Tuple[List[Tuple[int, int]], float]:
    """Return one inclusive-exclusive frame span for every target token."""

    emissions = log_probs.detach().float().cpu()
    target = torch.as_tensor(targets, dtype=torch.long).cpu()
    if emissions.ndim != 2:
        raise ValueError(f"Expected [time, vocab] log-probs, got {emissions.shape}")
    if target.ndim != 1 or target.numel() == 0:
        raise ValueError("CTC forced alignment requires a non-empty 1-D target")
    time_steps, vocab_size = emissions.shape
    if target.min().item() < 0 or target.max().item() >= vocab_size:
        raise ValueError("Target token is outside the emission vocabulary")

    num_states = 2 * target.numel() + 1
    extended = torch.full((num_states,), blank_id, dtype=torch.long)
    extended[1::2] = target
    can_skip = torch.zeros(num_states, dtype=torch.bool)
    if num_states > 2:
        can_skip[2:] = (extended[2:] != blank_id) & (
            extended[2:] != extended[:-2]
        )

    negative_inf = torch.tensor(float("-inf"), dtype=emissions.dtype)
    previous = torch.full((num_states,), negative_inf, dtype=emissions.dtype)
    previous[0] = emissions[0, blank_id]
    previous[1] = emissions[0, int(extended[1])]
    backpointers = torch.zeros((time_steps, num_states), dtype=torch.int8)

    for frame in range(1, time_steps):
        stay = previous
        step = torch.full_like(previous, negative_inf)
        step[1:] = previous[:-1]
        skip = torch.full_like(previous, negative_inf)
        skip[2:] = previous[:-2]
        skip = torch.where(can_skip, skip, negative_inf)
        best, transition = torch.stack((stay, step, skip), dim=0).max(dim=0)
        previous = best + emissions[frame, extended]
        backpointers[frame] = transition.to(torch.int8)

    final_states = [num_states - 1, num_states - 2]
    final_scores = torch.stack([previous[state] for state in final_states])
    best_final = int(final_scores.argmax().item())
    state = final_states[best_final]
    score = float(final_scores[best_final].item())
    if not math.isfinite(score):
        raise ValueError(
            f"No valid CTC path: time={time_steps}, targets={target.numel()}"
        )

    state_path = [0] * time_steps
    state_path[-1] = state
    for frame in range(time_steps - 1, 0, -1):
        state -= int(backpointers[frame, state].item())
        state_path[frame - 1] = state

    token_spans: List[Tuple[int, int]] = []
    for token_index in range(target.numel()):
        token_state = 2 * token_index + 1
        frames = [
            frame for frame, path_state in enumerate(state_path)
            if path_state == token_state
        ]
        if not frames:
            raise ValueError(f"Target token {token_index} has no aligned frame")
        token_spans.append((frames[0], frames[-1] + 1))
    return token_spans, score


def word_token_ranges(
    text: str,
    token_ids: Sequence[int],
    sentencepiece,
) -> List[Tuple[int, int]]:
    """Map transcript words to their exact SentencePiece token ranges."""

    words = text.split()
    per_word = [sentencepiece.encode_as_ids(word) for word in words]
    flattened = [token for word_ids in per_word for token in word_ids]
    if flattened != list(token_ids):
        raise ValueError(
            "SentencePiece full-sentence and per-word tokenization disagree: "
            f"{text!r}"
        )
    ranges = []
    begin = 0
    for word_ids in per_word:
        end = begin + len(word_ids)
        if end == begin:
            raise ValueError(f"Word produced no tokens: {words[len(ranges)]!r}")
        ranges.append((begin, end))
        begin = end
    return ranges


def token_spans_to_words(
    token_spans: Sequence[Tuple[int, int]],
    token_ranges: Sequence[Tuple[int, int]],
    seconds_per_frame: float,
    duration: float,
) -> List[Tuple[float, float]]:
    word_spans = []
    for token_begin, token_end in token_ranges:
        start = token_spans[token_begin][0] * seconds_per_frame
        end = token_spans[token_end - 1][1] * seconds_per_frame
        start = min(max(start, 0.0), duration)
        end = min(max(end, start), duration)
        word_spans.append((start, end))
    return word_spans


def load_mfa_manifest(mfa_dir: Path, dataset: str) -> Dict[str, Dict[str, Any]]:
    path = mfa_dir / f"librispeech_mfa_{dataset}.jsonl.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    records: Dict[str, Dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            records[record["id"]] = record
    return records


def _reference_word_spans(record: Mapping[str, Any]) -> List[Tuple[str, float, float]]:
    spans = []
    for item in record["words"]:
        start = float(item["start"])
        end = start + float(item["duration"])
        spans.append((str(item["symbol"]), start, end))
    return spans


def _word_rows(
    dataset: str,
    cut_id: str,
    model_name: str,
    reference: Sequence[Tuple[str, float, float]],
    predicted: Sequence[Tuple[float, float]],
    path_score: float,
    num_frames: int,
) -> List[Dict[str, Any]]:
    if len(reference) != len(predicted):
        raise ValueError(f"Word count mismatch for {cut_id}")
    rows = []
    for index, ((word, ref_start, ref_end), (pred_start, pred_end)) in enumerate(
        zip(reference, predicted)
    ):
        intersection = max(0.0, min(ref_end, pred_end) - max(ref_start, pred_start))
        union = max(ref_end, pred_end) - min(ref_start, pred_start)
        start_error = pred_start - ref_start
        end_error = pred_end - ref_end
        ref_center = 0.5 * (ref_start + ref_end)
        pred_center = 0.5 * (pred_start + pred_end)
        ref_duration = ref_end - ref_start
        pred_duration = pred_end - pred_start
        rows.append(
            {
                "dataset": dataset,
                "cut_id": cut_id,
                "model": model_name,
                "word_index": index,
                "word": word,
                "ref_start": ref_start,
                "ref_end": ref_end,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "start_error_ms": 1000.0 * start_error,
                "end_error_ms": 1000.0 * end_error,
                "start_abs_error_ms": 1000.0 * abs(start_error),
                "end_abs_error_ms": 1000.0 * abs(end_error),
                "center_abs_error_ms": 1000.0 * abs(pred_center - ref_center),
                "ref_duration_ms": 1000.0 * ref_duration,
                "pred_duration_ms": 1000.0 * pred_duration,
                "duration_error_ms": 1000.0 * (pred_duration - ref_duration),
                "duration_abs_error_ms": 1000.0
                * abs(pred_duration - ref_duration),
                "word_iou": intersection / union if union > 0.0 else 1.0,
                "path_score_per_frame": path_score / max(num_frames, 1),
            }
        )
    return rows


def _describe(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "std": float(values.std()),
    }


def _utterance_macro_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    metric_fns = {
        "boundary_abs_error_ms": lambda row: 0.5
        * (row["start_abs_error_ms"] + row["end_abs_error_ms"]),
        "start_abs_error_ms": lambda row: row["start_abs_error_ms"],
        "end_abs_error_ms": lambda row: row["end_abs_error_ms"],
        "center_abs_error_ms": lambda row: row["center_abs_error_ms"],
        "duration_abs_error_ms": lambda row: row["duration_abs_error_ms"],
        "pred_duration_ms": lambda row: row["pred_duration_ms"],
        "ref_duration_ms": lambda row: row["ref_duration_ms"],
        "word_iou": lambda row: row["word_iou"],
    }
    by_cut: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cut[str(row["cut_id"])].append(row)
    return {
        metric: _describe(
            np.asarray(
                [
                    np.mean([function(row) for row in cut_rows])
                    for cut_rows in by_cut.values()
                ],
                dtype=float,
            )
        )
        for metric, function in metric_fns.items()
    }


def _paired_summary(
    rows: Sequence[Mapping[str, Any]],
    candidate_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    metric_fns = {
        "boundary_abs_error_ms": lambda row: 0.5
        * (row["start_abs_error_ms"] + row["end_abs_error_ms"]),
        "center_abs_error_ms": lambda row: row["center_abs_error_ms"],
        "duration_abs_error_ms": lambda row: row["duration_abs_error_ms"],
        "word_iou": lambda row: row["word_iou"],
    }
    lower_is_better = {
        "boundary_abs_error_ms",
        "center_abs_error_ms",
        "duration_abs_error_ms",
    }
    datasets = sorted({str(row["dataset"]) for row in rows})
    rng = np.random.default_rng(bootstrap_seed)
    output: Dict[str, Any] = {}
    for dataset in datasets + ["combined"]:
        selected = [
            row for row in rows
            if dataset == "combined" or row["dataset"] == dataset
        ]
        by_word: Dict[Tuple[str, str, int], Dict[str, Mapping[str, Any]]] = (
            defaultdict(dict)
        )
        for row in selected:
            key = (str(row["dataset"]), str(row["cut_id"]), int(row["word_index"]))
            by_word[key][str(row["model"])] = row

        utterance_deltas: Dict[str, Dict[Tuple[str, str], List[float]]] = {
            metric: defaultdict(list) for metric in metric_fns
        }
        for (row_dataset, cut_id, _), model_rows in by_word.items():
            if "baseline" not in model_rows or candidate_name not in model_rows:
                continue
            for metric, function in metric_fns.items():
                delta = function(model_rows[candidate_name]) - function(
                    model_rows["baseline"]
                )
                utterance_deltas[metric][(row_dataset, cut_id)].append(float(delta))

        info: Dict[str, Any] = {"candidate": candidate_name, "metrics": {}}
        for metric, grouped in utterance_deltas.items():
            values = np.asarray(
                [np.mean(word_deltas) for word_deltas in grouped.values()], dtype=float
            )
            if values.size == 0:
                continue
            indices = rng.integers(
                0,
                values.size,
                size=(bootstrap_samples, values.size),
                dtype=np.int32,
            )
            boot_means = values[indices].mean(axis=1)
            if metric in lower_is_better:
                win_rate = float((values < 0.0).mean())
            else:
                win_rate = float((values > 0.0).mean())
            info["metrics"][metric] = {
                "candidate_minus_baseline": float(values.mean()),
                "ci95_low": float(np.quantile(boot_means, 0.025)),
                "ci95_high": float(np.quantile(boot_means, 0.975)),
                "utterance_win_rate": win_rate,
                "num_paired_utterances": int(values.size),
            }
        output[dataset] = info
    return output


def summarize(
    rows: Sequence[Mapping[str, Any]],
    candidate_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows})
    summary: Dict[str, Any] = {"datasets": {}, "models": models}
    for dataset in datasets + ["combined"]:
        summary["datasets"][dataset] = {}
        for model in models:
            selected = [
                row for row in rows
                if row["model"] == model
                and (dataset == "combined" or row["dataset"] == dataset)
            ]
            if not selected:
                continue
            start_abs = np.asarray(
                [row["start_abs_error_ms"] for row in selected], dtype=float
            )
            end_abs = np.asarray(
                [row["end_abs_error_ms"] for row in selected], dtype=float
            )
            boundary_abs = np.concatenate((start_abs, end_abs))
            metrics = {
                "start_abs_error_ms": _describe(start_abs),
                "end_abs_error_ms": _describe(end_abs),
                "boundary_abs_error_ms": _describe(boundary_abs),
                "center_abs_error_ms": _describe(
                    np.asarray([row["center_abs_error_ms"] for row in selected])
                ),
                "duration_abs_error_ms": _describe(
                    np.asarray([row["duration_abs_error_ms"] for row in selected])
                ),
                "ref_duration_ms": _describe(
                    np.asarray([row["ref_duration_ms"] for row in selected])
                ),
                "pred_duration_ms": _describe(
                    np.asarray([row["pred_duration_ms"] for row in selected])
                ),
                "duration_signed_error_ms": _describe(
                    np.asarray([row["duration_error_ms"] for row in selected])
                ),
                "word_iou": _describe(
                    np.asarray([row["word_iou"] for row in selected])
                ),
                "start_signed_error_ms": _describe(
                    np.asarray([row["start_error_ms"] for row in selected])
                ),
                "end_signed_error_ms": _describe(
                    np.asarray([row["end_error_ms"] for row in selected])
                ),
            }
            for tolerance in TOLERANCES_MS:
                metrics[f"boundary_within_{tolerance}ms"] = {
                    "mean": float((boundary_abs <= tolerance).mean())
                }
            summary["datasets"][dataset][model] = {
                "num_utterances": len({row["cut_id"] for row in selected}),
                "num_words": len(selected),
                "metrics": metrics,
                "utterance_macro_metrics": _utterance_macro_metrics(selected),
            }
    summary["pairwise"] = _paired_summary(
        rows=rows,
        candidate_name=candidate_name,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return summary


def write_summary_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# LibriSpeech MFA forced-alignment evaluation",
        "",
        "> MFA is an automatic pseudo-reference, not manual ground truth.",
        "",
        "| Split | Model | Utt. | Words | WBE, utt.-macro (ms) | Median (ms) | "
        "Within 80 ms | Onset MAE | Offset MAE | Pred. WDUR | Ref. WDUR | "
        "Word IoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, model_infos in summary["datasets"].items():
        for model, info in model_infos.items():
            metrics = info["metrics"]
            macro = info["utterance_macro_metrics"]
            boundary = macro["boundary_abs_error_ms"]
            lines.append(
                f"| {dataset} | {model} | {info['num_utterances']} | "
                f"{info['num_words']} | {boundary['mean']:.2f} | "
                f"{boundary['median']:.2f} | "
                f"{100.0 * metrics['boundary_within_80ms']['mean']:.2f}% | "
                f"{macro['start_abs_error_ms']['mean']:.2f} | "
                f"{macro['end_abs_error_ms']['mean']:.2f} | "
                f"{macro['pred_duration_ms']['mean']:.2f} | "
                f"{macro['ref_duration_ms']['mean']:.2f} | "
                f"{macro['word_iou']['mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Paired utterance bootstrap",
            "",
            "Deltas are candidate minus baseline; negative boundary MAE and positive "
            "word IoU are better.",
            "",
            "| Split | Paired utt. | Boundary MAE delta (ms) | 95% CI | "
            "Boundary win rate | Word IoU delta | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, info in summary["pairwise"].items():
        boundary = info["metrics"].get("boundary_abs_error_ms")
        word_iou = info["metrics"].get("word_iou")
        if boundary is None or word_iou is None:
            continue
        lines.append(
            f"| {dataset} | {boundary['num_paired_utterances']} | "
            f"{boundary['candidate_minus_baseline']:+.2f} | "
            f"[{boundary['ci95_low']:+.2f}, {boundary['ci95_high']:+.2f}] | "
            f"{100.0 * boundary['utterance_win_rate']:.2f}% | "
            f"{word_iou['candidate_minus_baseline']:+.4f} | "
            f"[{word_iou['ci95_low']:+.4f}, {word_iou['ci95_high']:+.4f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No word-level rows were produced")
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def librispeech_utterance_id(cut) -> str:
    """Return the canonical LibriSpeech ID rather than Lhotse's cut ID.

    The prepared cut IDs have an extra uniqueness suffix (for example
    ``2086-149214-0000-0``); the supervision/recording ID is the ID used by
    LibriSpeech and the MFA manifests.
    """

    if getattr(cut, "supervisions", None):
        return str(cut.supervisions[0].recording_id)
    return str(cut.recording_id)


def evaluate(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    lexicon = Lexicon(args.lang_dir)
    num_classes = max(lexicon.tokens) + 1
    graph = BpeCtcTrainingGraphCompiler(
        args.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )
    baseline = _build_baseline_model(args, num_classes, device)
    candidate = _build_evaluation_candidate(args, num_classes, device)
    seconds_per_frame = (
        args.frame_shift_ms * args.subsampling_factor / 1000.0
    )

    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for dataset in args.datasets:
        references = load_mfa_manifest(args.mfa_dir, dataset)
        args.dataset = dataset
        args.max_cuts = args.max_cuts_per_dataset
        dataloader = _load_eval_dataloader(args)
        processed = 0
        for batch_index, batch in enumerate(dataloader):
            baseline_outputs = _batch_log_probs(
                baseline,
                batch,
                device,
                ctc_nonblank_logit_bias=args.baseline_ctc_nonblank_logit_bias,
            )
            candidate_outputs = _batch_log_probs(
                candidate,
                batch,
                device,
                ctc_nonblank_logit_bias=args.candidate_ctc_nonblank_logit_bias,
            )
            if args.candidate_kind == "label_prior":
                # Label-prior scores are not normalized. Frame-wise
                # normalization leaves every CTC path ranking unchanged and
                # makes downstream probability diagnostics well defined.
                candidate_outputs = [
                    output.log_softmax(dim=-1) for output in candidate_outputs
                ]
            texts: Sequence[str] = batch["supervisions"]["text"]
            cuts = batch["supervisions"]["cut"]
            token_ids = graph.texts_to_ids(list(texts))

            for index, text in enumerate(texts):
                cut_id = librispeech_utterance_id(cuts[index])
                record = references.get(cut_id)
                if record is None or not record.get("words"):
                    skipped[dataset].append(
                        {"cut_id": cut_id, "reason": "missing_mfa_words"}
                    )
                    continue
                reference = _reference_word_spans(record)
                if [word for word, _, _ in reference] != text.split():
                    skipped[dataset].append(
                        {"cut_id": cut_id, "reason": "transcript_mismatch"}
                    )
                    continue
                try:
                    ranges = word_token_ranges(text, token_ids[index], graph.sp)
                    for model_name, log_probs in (
                        ("baseline", baseline_outputs[index]),
                        (args.candidate_name, candidate_outputs[index]),
                    ):
                        token_spans, path_score = ctc_viterbi_align(
                            log_probs, token_ids[index], blank_id=0
                        )
                        predicted = token_spans_to_words(
                            token_spans=token_spans,
                            token_ranges=ranges,
                            seconds_per_frame=seconds_per_frame,
                            duration=float(record["duration"]),
                        )
                        rows.extend(
                            _word_rows(
                                dataset=dataset,
                                cut_id=cut_id,
                                model_name=model_name,
                                reference=reference,
                                predicted=predicted,
                                path_score=path_score,
                                num_frames=log_probs.size(0),
                            )
                        )
                except ValueError as error:
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
    return rows, dict(skipped)


def main() -> None:
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(f"{args.output_dir}/log-mfa-alignment")
    logging.info("Arguments: %s", vars(args))
    rows, skipped = evaluate(args)
    summary = summarize(
        rows,
        candidate_name=args.candidate_name,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["reference"] = {
        "name": "LibriSpeech MFA alignments",
        "type": "automatic pseudo-reference",
        "mfa_dir": str(args.mfa_dir),
    }
    summary["configuration"] = {
        "baseline_exp_dir": str(args.baseline_exp_dir),
        "candidate_exp_dir": str(args.candidate_exp_dir),
        "candidate_kind": args.candidate_kind,
        "baseline_epoch": args.baseline_epoch,
        "candidate_epoch": args.candidate_epoch,
        "baseline_avg": args.baseline_avg,
        "candidate_avg": args.candidate_avg,
        "candidate_prior_logit_bias": args.candidate_prior_logit_bias,
        "candidate_ctc_nonblank_logit_bias": (
            args.candidate_ctc_nonblank_logit_bias
        ),
        "candidate_label_prior_alpha": args.candidate_label_prior_alpha,
        "candidate_label_prior_floor": args.candidate_label_prior_floor,
        "baseline_ctc_nonblank_logit_bias": args.baseline_ctc_nonblank_logit_bias,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "seconds_per_output_frame": (
            args.frame_shift_ms * args.subsampling_factor / 1000.0
        ),
    }
    write_csv(rows, args.output_dir / "word_metrics.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "skipped.json").write_text(
        json.dumps(skipped, indent=2) + "\n", encoding="utf-8"
    )
    write_summary_markdown(summary, args.output_dir / "summary.md")
    logging.info("Wrote MFA alignment evaluation to %s", args.output_dir)


if __name__ == "__main__":
    main()
