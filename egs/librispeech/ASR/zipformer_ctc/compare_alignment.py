#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from lhotse import CutSet
from lhotse.dataset import (
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
from lhotse import Fbank, FbankConfig
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint
from icefall.lexicon import Lexicon
from icefall.utils import AttributeDict, setup_logger
from shared_alignment_viz import compute_alignment_stats, save_alignment_report
from train import get_ctc_model, get_params


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--ot-checkpoint", type=Path, required=True)
    parser.add_argument("--lang-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="dev-clean")
    parser.add_argument("--cut-id", type=str, default=None)
    parser.add_argument("--cut-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("zipformer_ctc/alignment_compare"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ot-tau", type=float, default=0.1)
    parser.add_argument("--ot-eps", type=float, default=0.5)
    parser.add_argument("--ot-iters", type=int, default=40)
    parser.add_argument("--ot-beta-pos", type=float, default=1.0)
    parser.add_argument("--support-relative-threshold", type=float, default=0.1)
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
    params.vocab_size = max(lexicon.tokens) + 1
    model = get_ctc_model(params)
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


def _load_single_cut(args: argparse.Namespace):
    args.return_cuts = True
    args.shuffle = False
    args.drop_last = False
    args.enable_spec_aug = False
    args.enable_musan = False
    args.num_workers = 0
    args.max_duration = max(float(args.max_duration), 2000.0)

    datamodule = LibriSpeechAsrDataModule(args)
    cuts_fn_name = f"{args.dataset.replace('-', '_')}_cuts"
    if not hasattr(datamodule, cuts_fn_name):
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cuts = getattr(datamodule, cuts_fn_name)()
    selected_cut = None
    if args.cut_id is not None:
        for cut in cuts:
            supervision_ids = [sup.id for sup in getattr(cut, "supervisions", [])]
            if (
                cut.id == args.cut_id
                or args.cut_id in supervision_ids
                or cut.id.startswith(f"{args.cut_id}-")
            ):
                selected_cut = cut
                break
        if selected_cut is None:
            raise ValueError(
                f"Cut/supervision id {args.cut_id} was not found in {args.dataset}"
            )
    else:
        for idx, cut in enumerate(cuts):
            if idx == args.cut_index:
                selected_cut = cut
                break
        if selected_cut is None:
            raise ValueError(f"Cut index {args.cut_index} is out of range")

    single_cut = CutSet.from_cuts([selected_cut])
    if args.on_the_fly_feats:
        dataset = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            return_cuts=args.return_cuts,
        )
    else:
        if args.input_strategy != "PrecomputedFeatures":
            raise ValueError(
                f"Unsupported input_strategy for single-cut mode: {args.input_strategy}"
            )
        dataset = K2SpeechRecognitionDataset(
            input_strategy=PrecomputedFeatures(),
            return_cuts=args.return_cuts,
        )
    sampler = SimpleCutSampler(single_cut, max_duration=args.max_duration, shuffle=False)
    dl = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    batch = next(iter(dl))
    return selected_cut, batch


def _forward_log_probs(
    model,
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, str]:
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    with torch.no_grad():
        nnet_output, encoder_out_lens = model.encoder(feature, feature_lens)
        log_probs = model.ctc_output(nnet_output)

    output_len = max(int(encoder_out_lens[0].item()), 1)
    text = supervisions["text"][0]
    return log_probs[0, :output_len].detach().cpu(), text


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    setup_logger(f"{args.output_dir}/log-compare-alignment")

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    selected_cut, batch = _load_single_cut(args)

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

    baseline_log_probs, text = _forward_log_probs(baseline_model, batch, device)
    ot_log_probs, _ = _forward_log_probs(ot_model, batch, device)

    reports = []
    for name, log_probs, graph in [
        ("Baseline", baseline_log_probs, baseline_graph),
        ("OT-Trained", ot_log_probs, ot_graph),
    ]:
        label_ids = graph.texts_to_ids([text])[0]
        labels = torch.tensor(label_ids, dtype=torch.long)
        token_pieces = [graph.sp.id_to_piece(token_id) for token_id in label_ids]
        stats = compute_alignment_stats(
            log_probs=log_probs,
            labels=labels,
            token_pieces=token_pieces,
            blank_id=0,
            tau=args.ot_tau,
            eps=args.ot_eps,
            iters=args.ot_iters,
            beta_pos=args.ot_beta_pos,
            support_relative_threshold=args.support_relative_threshold,
        )
        reports.append(
            {
                "name": name,
                "cut_id": selected_cut.id,
                "text": text,
                "stats": stats,
            }
        )

    safe_cut_id = selected_cut.id.replace("/", "_")
    out_prefix = args.output_dir / f"{safe_cut_id}_alignment_compare"
    title = f"{selected_cut.id} | {text}"
    outputs = save_alignment_report(reports=reports, out_prefix=out_prefix, title=title)

    logging.info("Saved figure to %s", outputs["png"])
    logging.info("Saved tensors to %s", outputs["pt"])


if __name__ == "__main__":
    main()
