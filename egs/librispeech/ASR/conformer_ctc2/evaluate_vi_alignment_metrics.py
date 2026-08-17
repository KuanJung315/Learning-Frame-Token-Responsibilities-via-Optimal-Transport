#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent
if str(ASR_DIR) not in sys.path:
    sys.path.insert(0, str(ASR_DIR))

from asr_datamodule import LibriSpeechAsrDataModule
from blank_gate_v2 import BlankGateHeadV2, BlankPriorHeadV2
from conformer import Conformer
from evaluate_alignment_metrics import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    _checkpoint_metadata,
    _load_eval_dataloader,
)
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import setup_logger, str2bool
from shared_alignment_viz import (
    compute_alignment_quality_metrics,
    compute_alignment_stats,
)
from train import get_params as get_baseline_params
from varctc_v2_utils import build_gated_log_probs_v2, encoder_lens_from_mask


class ConformerVIV2ForAlignment(nn.Module):
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

    def alignment_forward(self, x, supervisions=None, warmup: float = 1.0):
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
        log_probs = build_gated_log_probs_v2(log_p_nonblank, alpha_prior)
        return (
            log_probs,
            log_p_nonblank,
            alpha_prior,
            encoder_out_lens,
            memory,
            memory_key_padding_mask,
        )

    def forward(self, x, supervisions=None, warmup: float = 1.0):
        outputs = self.alignment_forward(x, supervisions, warmup=warmup)
        log_probs, memory, memory_key_padding_mask = outputs[0], outputs[4], outputs[5]
        return log_probs, memory, memory_key_padding_mask

    def decoder_forward(self, *args, **kwargs):
        return self.encoder.decoder_forward(*args, **kwargs)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--baseline-exp-dir", type=Path, required=True)
    parser.add_argument("--baseline-epoch", type=int, default=30)
    parser.add_argument("--baseline-avg", type=int, default=10)
    parser.add_argument("--baseline-use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--baseline-num-decoder-layers", type=int, default=6)
    parser.add_argument(
        "--baseline-ctc-nonblank-logit-bias",
        type=float,
        default=0.0,
        help=(
            "Apply an equivalent baseline CTC calibration before computing "
            "alignment metrics: add this value to all non-blank log-probs and "
            "renormalize."
        ),
    )

    parser.add_argument("--candidate-exp-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", type=str, default="varctc_v2")
    parser.add_argument("--candidate-epoch", type=int, default=30)
    parser.add_argument("--candidate-avg", type=int, default=10)
    parser.add_argument("--candidate-use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--candidate-num-decoder-layers", type=int, default=6)
    parser.add_argument("--candidate-prior-logit-bias", type=float, default=0.0)
    parser.add_argument("--label-embed-dim", type=int, default=256)
    parser.add_argument("--init-blank-prob", type=float, default=0.35)

    parser.add_argument("--lang-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="dev-other")
    parser.add_argument("--max-cuts", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/alignment_eval_vi_v2"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ot-tau", type=float, default=0.1)
    parser.add_argument("--ot-eps", type=float, default=0.3)
    parser.add_argument("--ot-iters", type=int, default=30)
    parser.add_argument("--ot-beta-pos", type=float, default=1.0)
    parser.add_argument("--support-relative-threshold", type=float, default=0.1)
    parser.add_argument("--diagonal-band-width", type=float, default=0.12)
    parser.add_argument("--backward-tol", type=float, default=0.05)
    LibriSpeechAsrDataModule.add_arguments(parser)
    return parser


def _checkpoint_path(exp_dir: Path, epoch: int) -> Path:
    path = exp_dir / f"epoch-{epoch}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return path


def _load_state(
    model: nn.Module,
    exp_dir: Path,
    epoch: int,
    avg: int,
    use_averaged_model: bool,
    device: torch.device,
) -> None:
    model.to(device)
    if use_averaged_model:
        start = epoch - avg
        if start < 1:
            raise ValueError(
                f"Cannot use averaged model with epoch={epoch}, avg={avg}; "
                "epoch-avg must be >= 1."
            )
        state_dict = average_checkpoints_with_averaged_model(
            filename_start=str(_checkpoint_path(exp_dir, start)),
            filename_end=str(_checkpoint_path(exp_dir, epoch)),
            device=device,
        )
        model.load_state_dict(state_dict)
    elif avg == 1:
        load_checkpoint(str(_checkpoint_path(exp_dir, epoch)), model=model)
    else:
        start = epoch - avg + 1
        filenames = [str(_checkpoint_path(exp_dir, i)) for i in range(start, epoch + 1)]
        model.load_state_dict(average_checkpoints(filenames, device=device))
    model.eval()


def _build_baseline_model(
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> Conformer:
    saved = _checkpoint_metadata(_checkpoint_path(args.baseline_exp_dir, args.baseline_epoch))
    params = get_baseline_params()
    params.update(saved)
    if not hasattr(params, "num_decoder_layers"):
        params.num_decoder_layers = args.baseline_num_decoder_layers

    model = Conformer(
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
        exp_dir=args.baseline_exp_dir,
        epoch=args.baseline_epoch,
        avg=args.baseline_avg,
        use_averaged_model=args.baseline_use_averaged_model,
        device=device,
    )
    return model


def _build_candidate_model(
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> ConformerVIV2ForAlignment:
    saved = _checkpoint_metadata(_checkpoint_path(args.candidate_exp_dir, args.candidate_epoch))
    params = get_baseline_params()
    params.update(saved)
    if not hasattr(params, "num_decoder_layers"):
        params.num_decoder_layers = args.candidate_num_decoder_layers
    if not hasattr(params, "label_embed_dim"):
        params.label_embed_dim = args.label_embed_dim
    if not hasattr(params, "init_blank_prob"):
        params.init_blank_prob = args.init_blank_prob

    encoder = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )
    for p in encoder.encoder_output_layer.parameters():
        p.requires_grad_(False)

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
    model = ConformerVIV2ForAlignment(
        encoder=encoder,
        ctc_head=ctc_head,
        blank_gate=blank_gate,
        blank_prior=blank_prior,
        prior_logit_bias=args.candidate_prior_logit_bias,
    )
    _load_state(
        model=model,
        exp_dir=args.candidate_exp_dir,
        epoch=args.candidate_epoch,
        avg=args.candidate_avg,
        use_averaged_model=args.candidate_use_averaged_model,
        device=device,
    )
    return model


def _apply_ctc_nonblank_logit_bias(
    log_probs: torch.Tensor,
    nonblank_bias: float,
) -> torch.Tensor:
    if nonblank_bias == 0.0:
        return log_probs
    biased = log_probs.clone()
    biased[..., 1:] = biased[..., 1:] + nonblank_bias
    return biased - biased.logsumexp(dim=-1, keepdim=True)


def _batch_log_probs(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    ctc_nonblank_logit_bias: float = 0.0,
) -> List[torch.Tensor]:
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    with torch.no_grad():
        log_probs, _, _ = model(feature, supervisions, warmup=1.0)
        log_probs = _apply_ctc_nonblank_logit_bias(
            log_probs, ctc_nonblank_logit_bias
        )

    outputs = []
    for i in range(log_probs.size(0)):
        num_frames = int(supervisions["num_frames"][i].item())
        output_len = ((num_frames - 1) // 2 - 1) // 2
        output_len = max(output_len, 1)
        outputs.append(log_probs[i, :output_len].detach().cpu())
    return outputs


def _collect_rows(
    dl,
    baseline_model: nn.Module,
    candidate_model: nn.Module,
    graph: BpeCtcTrainingGraphCompiler,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(dl):
        baseline_outputs = _batch_log_probs(
            baseline_model,
            batch,
            device,
            ctc_nonblank_logit_bias=args.baseline_ctc_nonblank_logit_bias,
        )
        candidate_outputs = _batch_log_probs(candidate_model, batch, device)
        texts: Sequence[str] = batch["supervisions"]["text"]
        cuts = batch["supervisions"].get("cut", [None] * len(texts))
        token_ids = graph.texts_to_ids(list(texts))

        for i, text in enumerate(texts):
            cut = cuts[i] if i < len(cuts) else None
            cut_id = cut.id if cut is not None else f"batch{batch_idx}_utt{i}"
            labels = torch.tensor(token_ids[i], dtype=torch.long)
            pieces = [graph.sp.id_to_piece(token_id) for token_id in token_ids[i]]

            baseline_stats = compute_alignment_stats(
                log_probs=baseline_outputs[i],
                labels=labels,
                token_pieces=pieces,
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

            candidate_stats = compute_alignment_stats(
                log_probs=candidate_outputs[i],
                labels=labels,
                token_pieces=pieces,
                blank_id=0,
                tau=args.ot_tau,
                eps=args.ot_eps,
                iters=args.ot_iters,
                beta_pos=args.ot_beta_pos,
                support_relative_threshold=args.support_relative_threshold,
            )
            candidate_metrics = compute_alignment_quality_metrics(
                candidate_stats,
                diagonal_band_width=args.diagonal_band_width,
                backward_tol=args.backward_tol,
            )

            row: Dict[str, Any] = {
                "cut_id": cut_id,
                "text": text,
                "num_tokens": len(token_ids[i]),
            }
            for key, value in baseline_metrics.items():
                row[f"baseline__{key}"] = value
            for key, value in candidate_metrics.items():
                row[f"candidate__{key}"] = value
                row[f"delta__{key}"] = value - baseline_metrics[key]
            rows.append(row)
    return rows


def _metric_names(rows: List[Dict[str, Any]]) -> List[str]:
    return sorted(key.split("__", 1)[1] for key in rows[0] if key.startswith("baseline__"))


def _aggregate_summary(
    rows: List[Dict[str, Any]],
    metric_names: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_utterances": len(rows),
        "dataset": args.dataset,
        "baseline": {
            "exp_dir": str(args.baseline_exp_dir),
            "epoch": args.baseline_epoch,
            "avg": args.baseline_avg,
            "use_averaged_model": args.baseline_use_averaged_model,
            "ctc_nonblank_logit_bias": args.baseline_ctc_nonblank_logit_bias,
        },
        "candidate": {
            "name": args.candidate_name,
            "exp_dir": str(args.candidate_exp_dir),
            "epoch": args.candidate_epoch,
            "avg": args.candidate_avg,
            "use_averaged_model": args.candidate_use_averaged_model,
            "prior_logit_bias": args.candidate_prior_logit_bias,
        },
        "metrics": {},
    }
    for metric in metric_names:
        baseline_vals = np.asarray([row[f"baseline__{metric}"] for row in rows], dtype=float)
        candidate_vals = np.asarray(
            [row[f"candidate__{metric}"] for row in rows], dtype=float
        )
        delta_vals = candidate_vals - baseline_vals

        direction = "neutral"
        if metric in LOWER_IS_BETTER:
            direction = "lower_is_better"
        elif metric in HIGHER_IS_BETTER:
            direction = "higher_is_better"

        summary["metrics"][metric] = {
            "direction": direction,
            "baseline_mean": float(baseline_vals.mean()),
            "candidate_mean": float(candidate_vals.mean()),
            "delta_mean": float(delta_vals.mean()),
            "baseline_median": float(np.median(baseline_vals)),
            "candidate_median": float(np.median(candidate_vals)),
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
        "candidate",
        "delta",
        "direction",
    )
    for metric in metric_names:
        info = summary["metrics"][metric]
        logging.info(
            "%-24s %12.5f %12.5f %12.5f %18s",
            metric,
            info["baseline_mean"],
            info["candidate_mean"],
            info["delta_mean"],
            info["direction"],
        )


def _bias_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(f"{args.output_dir}/log-evaluate-vi-alignment")
    logging.info("Alignment evaluation started")
    logging.info(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dl = _load_eval_dataloader(args)
    lexicon = Lexicon(args.lang_dir)
    num_classes = max(lexicon.tokens) + 1
    graph = BpeCtcTrainingGraphCompiler(
        args.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )

    baseline_model = _build_baseline_model(args, num_classes=num_classes, device=device)
    candidate_model = _build_candidate_model(args, num_classes=num_classes, device=device)

    rows = _collect_rows(
        dl=dl,
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        graph=graph,
        device=device,
        args=args,
    )
    if not rows:
        raise ValueError("No utterances were evaluated.")

    metric_names = _metric_names(rows)
    summary = _aggregate_summary(rows, metric_names, args)

    prefix = (
        args.output_dir
        / f"{args.dataset.replace('/', '_')}_"
        f"basebias{_bias_tag(args.baseline_ctc_nonblank_logit_bias)}_"
        f"{args.candidate_name}_e{args.candidate_epoch}_avg{args.candidate_avg}_"
        f"bias{_bias_tag(args.candidate_prior_logit_bias)}"
    )
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
