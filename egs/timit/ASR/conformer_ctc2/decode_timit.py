#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from timit_eval_common import (
    add_data_arguments,
    baseline_batch_outputs,
    build_baseline_model,
    build_phone_graph,
    build_vi_model,
    get_split_dataloader,
    greedy_runs,
    resolve_device,
    token_ids_to_symbols,
    vi_batch_outputs,
)

from icefall.utils import setup_logger, store_transcripts, write_error_stats


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("conformer_ctc2/exp_baseline_phone_small/epoch-50.pt"),
    )
    parser.add_argument(
        "--vi-checkpoint",
        type=Path,
        default=Path("conformer_ctc2/exp_vi_ot_phone_small/epoch-50.pt"),
    )
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_phone"))
    parser.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    parser.add_argument("--max-cuts", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vi-prior-logit-bias", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_ctc2/timit_decode"),
    )
    add_data_arguments(parser)
    return parser


def decode_split(
    dl,
    baseline_model,
    vi_model,
    graph,
    device,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    results: Dict[str, List[Tuple[str, List[str], List[str]]]] = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        baseline_outputs = baseline_batch_outputs(baseline_model, batch, device)
        vi_outputs, _ = vi_batch_outputs(
            vi_model, batch, graph=graph, device=device, gate="prior"
        )
        sequence_idx = batch["supervisions"]["sequence_idx"].tolist()
        cuts = batch["supervisions"]["cut"]
        texts = batch["supervisions"]["text"]
        for sup_idx, (cut, text) in enumerate(zip(cuts, texts)):
            seq_idx = int(sequence_idx[sup_idx])
            ref = text.split()
            for name, output in (
                ("baseline_greedy", baseline_outputs[seq_idx]),
                ("vi_prior_greedy", vi_outputs[seq_idx]),
            ):
                runs = greedy_runs(output.log_probs)
                hyp = token_ids_to_symbols(
                    [run["token_id"] for run in runs],
                    graph.token_table,
                )
                results[name].append((cut.id, ref, hyp))
        if batch_idx % 50 == 0:
            logging.info("decoded batch %s", batch_idx)
    return results


def save_split_results(
    results: Dict[str, List[Tuple[str, List[str], List[str]]]],
    split: str,
    output_dir: Path,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for name, rows in results.items():
        rows = sorted(rows)
        recog_path = output_dir / f"recogs-{split}-{name}.txt"
        errs_path = output_dir / f"errs-{split}-{name}.txt"
        store_transcripts(filename=recog_path, texts=rows)
        with open(errs_path, "w") as f:
            print("# Phone-token evaluation: the generic %WER value below is PER.", file=f)
            per = write_error_stats(
                f,
                test_set_name=f"{split}-{name}",
                results=rows,
                enable_log=True,
            )
        summary[name] = per
        logging.info("%s %s PER: %.2f", split, name, per)
    return summary


def main() -> None:
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(args.output_dir / "log-decode-timit"))
    device = resolve_device(args.device)
    logging.info("device: %s", device)

    _, graph = build_phone_graph(args.lang_dir, device=device)
    baseline_model, _ = build_baseline_model(
        args.baseline_checkpoint, args.lang_dir, device=device
    )
    vi_model, _ = build_vi_model(
        args.vi_checkpoint,
        args.lang_dir,
        device=device,
        prior_logit_bias=args.vi_prior_logit_bias,
    )

    all_summary = {}
    for split in args.splits:
        results = decode_split(
            get_split_dataloader(args, split),
            baseline_model=baseline_model,
            vi_model=vi_model,
            graph=graph,
            device=device,
        )
        all_summary[split] = save_split_results(results, split, args.output_dir)

    summary_path = args.output_dir / "per-summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2, sort_keys=True)
    logging.info("saved %s", summary_path)


if __name__ == "__main__":
    main()
