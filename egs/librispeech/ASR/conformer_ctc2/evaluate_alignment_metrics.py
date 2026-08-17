#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent
if str(ASR_DIR) not in sys.path:
    sys.path.insert(0, str(ASR_DIR))

from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint
from icefall.lexicon import Lexicon
from icefall.utils import AttributeDict, setup_logger
from shared_alignment_viz import (
    compute_alignment_quality_metrics,
    compute_alignment_stats,
)
from train import get_params


LOWER_IS_BETTER = {
    "mean_blank_prob",
    "mean_frame_entropy",
    "posterior_peakiness",
    "ctc_diag_mean_abs_dev",
    "ctc_offdiag_mass",
    "ot_diag_mean_abs_dev",
    "ot_offdiag_mass",
    "ctc_bary_jitter",
    "ctc_backward_rate",
    "ot_bary_jitter",
    "ot_backward_rate",
}

HIGHER_IS_BETTER = {
    "keep_ratio",
    "mean_nonblank_prob",
    "argmax_nonblank_ratio",
    "spike_width_mean",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--ot-checkpoint", type=Path, required=True)
    parser.add_argument("--lang-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="dev-other")
    parser.add_argument("--max-cuts", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/alignment_eval"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ot-tau", type=float, default=0.1)
    parser.add_argument("--ot-eps", type=float, default=0.5)
    parser.add_argument("--ot-iters", type=int, default=40)
    parser.add_argument("--ot-beta-pos", type=float, default=1.0)
    parser.add_argument("--support-relative-threshold", type=float, default=0.1)
    parser.add_argument("--diagonal-band-width", type=float, default=0.12)
    parser.add_argument("--backward-tol", type=float, default=0.05)
    LibriSpeechAsrDataModule.add_arguments(parser)
    return parser


def _checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ignored = {
        "model",
        "model_avg",
        "optimizer",
        "scheduler",
        "grad_scaler",
        "sampler",
    }
    return {k: v for k, v in checkpoint.items() if k not in ignored}


def _build_params(saved: Dict[str, Any], args: argparse.Namespace) -> AttributeDict:
    params = get_params()
    params.update(saved)
    params.update(vars(args))
    params.lang_dir = Path(args.lang_dir)
    return params


def _flag_was_set(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _load_model(checkpoint_path: Path, args: argparse.Namespace, device: torch.device):
    saved = _checkpoint_metadata(checkpoint_path)
    params = _build_params(saved, args)

    lexicon = Lexicon(params.lang_dir)
    num_classes = max(lexicon.tokens) + 1
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )
    load_checkpoint(checkpoint_path, model=model)
    model.to(device)
    model.eval()

    graph_compiler = BpeCtcTrainingGraphCompiler(
        params.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )
    return params, model, graph_compiler


def _take_first_cuts(cuts: Iterable, max_cuts: int) -> CutSet:
    if max_cuts <= 0:
        return CutSet.from_cuts(list(cuts))
    return CutSet.from_cuts(list(islice(cuts, max_cuts)))


def _load_eval_dataloader(args: argparse.Namespace):
    args.return_cuts = True
    args.shuffle = False
    args.drop_last = False
    args.enable_spec_aug = False
    args.enable_musan = False
    args.num_workers = 0

    datamodule = LibriSpeechAsrDataModule(args)
    cuts_fn_name = f"{args.dataset.replace('-', '_')}_cuts"
    if not hasattr(datamodule, cuts_fn_name):
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cuts = getattr(datamodule, cuts_fn_name)()
    subset_cuts = _take_first_cuts(cuts, args.max_cuts)

    if args.on_the_fly_feats:
        dataset = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            return_cuts=args.return_cuts,
        )
    else:
        if args.input_strategy != "PrecomputedFeatures":
            raise ValueError(
                f"Unsupported input_strategy for eval mode: {args.input_strategy}"
            )
        dataset = K2SpeechRecognitionDataset(
            input_strategy=PrecomputedFeatures(),
            return_cuts=args.return_cuts,
        )

    if len(subset_cuts) <= max(int(args.num_buckets), 1):
        sampler = SimpleCutSampler(
            subset_cuts,
            max_duration=args.max_duration,
            shuffle=False,
        )
    else:
        sampler = DynamicBucketingSampler(
            subset_cuts,
            max_duration=args.max_duration,
            shuffle=False,
            num_buckets=args.num_buckets,
        )

    dl = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=0,
        persistent_workers=False,
    )
    return dl


def _batch_log_probs(
    model: Conformer,
    batch: Dict[str, Any],
    device: torch.device,
) -> List[torch.Tensor]:
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    with torch.no_grad():
        log_probs, _, _ = model(feature, supervisions, warmup=1.0)

    outputs = []
    for i in range(log_probs.size(0)):
        num_frames = int(supervisions["num_frames"][i].item())
        output_len = ((num_frames - 1) // 2 - 1) // 2
        output_len = max(output_len, 1)
        outputs.append(log_probs[i, :output_len].detach().cpu())
    return outputs


def _collect_rows(
    dl,
    baseline_model: Conformer,
    baseline_graph: BpeCtcTrainingGraphCompiler,
    ot_model: Conformer,
    ot_graph: BpeCtcTrainingGraphCompiler,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(dl):
        baseline_outputs = _batch_log_probs(baseline_model, batch, device)
        ot_outputs = _batch_log_probs(ot_model, batch, device)
        texts: Sequence[str] = batch["supervisions"]["text"]
        cuts = batch["supervisions"].get("cut", [None] * len(texts))
        baseline_ids = baseline_graph.texts_to_ids(list(texts))
        ot_ids = ot_graph.texts_to_ids(list(texts))

        for i, text in enumerate(texts):
            cut = cuts[i] if i < len(cuts) else None
            cut_id = cut.id if cut is not None else f"batch{batch_idx}_utt{i}"

            baseline_labels = torch.tensor(baseline_ids[i], dtype=torch.long)
            baseline_pieces = [
                baseline_graph.sp.id_to_piece(token_id) for token_id in baseline_ids[i]
            ]
            baseline_stats = compute_alignment_stats(
                log_probs=baseline_outputs[i],
                labels=baseline_labels,
                token_pieces=baseline_pieces,
                blank_id=0,
                tau=args.ot_tau,
                eps=args.ot_eps,
                iters=args.ot_iters,
                beta_pos=args.ot_beta_pos,
                support_relative_threshold=args.support_relative_threshold,
            )
            baseline_metrics = compute_alignment_quality_metrics(
                baseline_stats,
                diagonal_band_width=args.diagonal_band_width,
                backward_tol=args.backward_tol,
            )

            ot_labels = torch.tensor(ot_ids[i], dtype=torch.long)
            ot_pieces = [ot_graph.sp.id_to_piece(token_id) for token_id in ot_ids[i]]
            ot_stats = compute_alignment_stats(
                log_probs=ot_outputs[i],
                labels=ot_labels,
                token_pieces=ot_pieces,
                blank_id=0,
                tau=args.ot_tau,
                eps=args.ot_eps,
                iters=args.ot_iters,
                beta_pos=args.ot_beta_pos,
                support_relative_threshold=args.support_relative_threshold,
            )
            ot_metrics = compute_alignment_quality_metrics(
                ot_stats,
                diagonal_band_width=args.diagonal_band_width,
                backward_tol=args.backward_tol,
            )

            row: Dict[str, Any] = {
                "cut_id": cut_id,
                "text": text,
                "num_tokens": len(baseline_ids[i]),
            }
            for key, value in baseline_metrics.items():
                row[f"baseline__{key}"] = value
            for key, value in ot_metrics.items():
                row[f"ot__{key}"] = value
                base_value = baseline_metrics.get(key)
                if base_value is not None:
                    row[f"delta__{key}"] = value - base_value
            rows.append(row)
    return rows


def _metric_names(rows: List[Dict[str, Any]]) -> List[str]:
    names = []
    for key in rows[0]:
        if key.startswith("baseline__"):
            names.append(key.split("__", 1)[1])
    return sorted(names)


def _aggregate_summary(rows: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_utterances": len(rows),
        "metrics": {},
    }
    for metric in metric_names:
        baseline_vals = np.asarray([row[f"baseline__{metric}"] for row in rows], dtype=float)
        ot_vals = np.asarray([row[f"ot__{metric}"] for row in rows], dtype=float)
        delta_vals = ot_vals - baseline_vals

        direction = "neutral"
        if metric in LOWER_IS_BETTER:
            direction = "lower_is_better"
        elif metric in HIGHER_IS_BETTER:
            direction = "higher_is_better"

        summary["metrics"][metric] = {
            "direction": direction,
            "baseline_mean": float(baseline_vals.mean()),
            "ot_mean": float(ot_vals.mean()),
            "delta_mean": float(delta_vals.mean()),
            "baseline_median": float(np.median(baseline_vals)),
            "ot_median": float(np.median(ot_vals)),
            "delta_median": float(np.median(delta_vals)),
        }
    return summary


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary: Dict[str, Any], metric_names: List[str]) -> None:
    logging.info("Alignment evaluation over %s utterances", summary["num_utterances"])
    logging.info(
        "%-24s %12s %12s %12s %18s",
        "metric",
        "baseline",
        "ot",
        "delta",
        "direction",
    )
    for metric in metric_names:
        info = summary["metrics"][metric]
        logging.info(
            "%-24s %12.5f %12.5f %12.5f %18s",
            metric,
            info["baseline_mean"],
            info["ot_mean"],
            info["delta_mean"],
            info["direction"],
        )


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(f"{args.output_dir}/log-evaluate-alignment")

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dl = _load_eval_dataloader(args)

    _, baseline_model, baseline_graph = _load_model(
        args.baseline_checkpoint, args=args, device=device
    )
    ot_params, ot_model, ot_graph = _load_model(
        args.ot_checkpoint, args=args, device=device
    )

    if not _flag_was_set("--ot-tau") and hasattr(ot_params, "ot_tau"):
        args.ot_tau = ot_params.ot_tau
    if not _flag_was_set("--ot-eps") and hasattr(ot_params, "ot_eps"):
        args.ot_eps = ot_params.ot_eps
    if not _flag_was_set("--ot-iters") and hasattr(ot_params, "ot_iters"):
        args.ot_iters = ot_params.ot_iters
    if not _flag_was_set("--ot-beta-pos") and hasattr(ot_params, "ot_beta_pos"):
        args.ot_beta_pos = ot_params.ot_beta_pos

    rows = _collect_rows(
        dl=dl,
        baseline_model=baseline_model,
        baseline_graph=baseline_graph,
        ot_model=ot_model,
        ot_graph=ot_graph,
        device=device,
        args=args,
    )
    if not rows:
        raise ValueError("No utterances were evaluated.")

    metric_names = _metric_names(rows)
    summary = _aggregate_summary(rows, metric_names)

    prefix = args.output_dir / f"{args.dataset.replace('/', '_')}_alignment_metrics"
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    _write_csv(rows, csv_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    _print_summary(summary, metric_names)
    logging.info("Saved per-utterance metrics to %s", csv_path)
    logging.info("Saved aggregate summary to %s", json_path)


if __name__ == "__main__":
    main()
