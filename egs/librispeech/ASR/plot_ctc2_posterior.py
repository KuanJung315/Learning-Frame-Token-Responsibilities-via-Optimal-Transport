#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR / "conformer_ctc2"
PROJECT_ROOT = SCRIPT_DIR.parents[2]

if str(RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(RECIPE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch

from conformer import Conformer
from icefall.lexicon import Lexicon
from icefall.utils import str2bool
from plot_vi_posterior import (
    _checkpoint_metadata,
    _checkpoint_path,
    _frame_to_x,
    _load_state,
    _make_x_values,
    _select_cut,
    _select_token_window,
    _word_token_ranges,
    ctc_collapse,
    ctc_forced_align_viterbi,
)
from train import get_params


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("conformer_ctc2/exp_baseline_li100h"),
    )
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--avg", type=int, default=10)
    parser.add_argument("--use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_bpe_500"))
    parser.add_argument(
        "--cuts-path",
        type=Path,
        default=Path("data/fbank/librispeech_cuts_test-clean.jsonl.gz"),
    )
    parser.add_argument("--cut-index", type=int, default=0)
    parser.add_argument("--cut-id", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--word-start", type=int, default=0)
    parser.add_argument("--num-words", type=int, default=8)
    parser.add_argument("--context-sec", type=float, default=0.25)
    parser.add_argument(
        "--label-mode",
        type=str,
        default="none",
        choices=["none", "words", "pieces"],
        help="Bottom labels on the posterior panel.",
    )
    parser.add_argument(
        "--top-label-mode",
        type=str,
        default="pieces",
        choices=["none", "words", "pieces"],
        help="Labels drawn above the spectrogram panel.",
    )
    parser.add_argument(
        "--top-label-stagger",
        type=str2bool,
        default=False,
        help="Stagger adjacent BPE labels above the spectrogram to reduce overlap.",
    )
    parser.add_argument(
        "--include-spectrogram",
        type=str2bool,
        default=True,
        help="Draw a log-fbank spectrogram panel above the posterior panel.",
    )
    parser.add_argument(
        "--show-blank",
        type=str2bool,
        default=False,
        help="Draw the CTC blank posterior as a dashed line.",
    )
    parser.add_argument(
        "--show-token-boundaries",
        type=str2bool,
        default=True,
        help="Draw light vertical guides at forced-aligned BPE token boundaries.",
    )
    parser.add_argument(
        "--show-legend",
        type=str2bool,
        default=False,
        help="Draw a legend for BPE posterior curves.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title. Leave empty for a cleaner paper-style plot.",
    )
    parser.add_argument(
        "--x-axis-unit",
        type=str,
        default="time",
        choices=["frame", "input_frame", "time"],
    )
    parser.add_argument(
        "--relative-x-axis",
        type=str2bool,
        default=False,
        help="Shift the plotted window so the left edge starts from 0.",
    )
    parser.add_argument(
        "--plot-full-cut",
        type=str2bool,
        default=False,
        help="Plot the whole cut from frame 0 instead of cropping to selected tokens.",
    )
    parser.add_argument(
        "--subsampling-center-offset",
        type=float,
        default=3.0,
        help=(
            "Input-frame offset for the center of each subsampled posterior frame. "
            "For the recipe's Conv2dSubsampling, output frame t is centered near "
            "input frame 4*t+3."
        ),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("conformer_ctc2/posterior_viz/baseline_ctc.png"),
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Optional PDF output for paper inclusion.",
    )
    parser.add_argument("--fig-width", type=float, default=16.0)
    parser.add_argument("--fig-height", type=float, default=3.6)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--line-width", type=float, default=1.65)
    parser.add_argument("--blank-line-width", type=float, default=1.5)
    parser.add_argument("--boundary-line-width", type=float, default=0.55)
    parser.add_argument("--axis-label-size", type=float, default=11.0)
    parser.add_argument("--tick-label-size", type=float, default=10.0)
    parser.add_argument("--token-label-size", type=float, default=8.0)
    parser.add_argument("--bottom-label-size", type=float, default=8.0)
    parser.add_argument("--spec-height-ratio", type=float, default=0.85)
    parser.add_argument("--post-height-ratio", type=float, default=1.35)
    parser.add_argument(
        "--y-bottom-pad",
        type=float,
        default=0.08,
        help="Extra probability-axis space below y=0 so curves do not touch the bottom frame.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def _piece_label(piece: str) -> str:
    return piece.replace("▁", "_")


def _token_color(idx: int) -> str:
    palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]
    return palette[idx % len(palette)]


def _input_frame_to_x(input_frame: float, x_axis_unit: str, subsampling: int) -> float:
    if x_axis_unit == "frame":
        return input_frame / subsampling
    if x_axis_unit == "input_frame":
        return input_frame
    return input_frame * 0.01


def _posterior_frame_to_x(
    frame: float,
    x_axis_unit: str,
    subsampling: int,
    center_offset: float,
) -> float:
    return _input_frame_to_x(
        frame * subsampling + center_offset,
        x_axis_unit,
        subsampling,
    )


def _plot_spectrogram(
    ax,
    feats: np.ndarray,
    frame_start: int,
    frame_end: int,
    x_axis_unit: str,
    subsampling: int,
    x_offset: float,
) -> None:
    input_start = max(0, int(frame_start) * subsampling)
    input_end = min(feats.shape[0], max(input_start + 1, int(frame_end) * subsampling))
    spec = feats[input_start:input_end].T
    lo, hi = np.percentile(spec, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    spec = np.clip((spec - lo) / (hi - lo), 0.0, 1.0)
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="RdYlBu_r",
        extent=[
            _input_frame_to_x(input_start, x_axis_unit, subsampling) - x_offset,
            _input_frame_to_x(input_end, x_axis_unit, subsampling) - x_offset,
            0,
            feats.shape[1],
        ],
    )
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def _draw_token_boundaries(
    axes,
    selected_segments,
    x_axis_unit: str,
    subsampling: int,
    center_offset: float,
    x_offset: float,
    linewidth: float,
) -> None:
    seen = set()
    for _, start, end in selected_segments:
        for frame in (start, end):
            if frame in seen:
                continue
            seen.add(frame)
            x = (
                _posterior_frame_to_x(frame, x_axis_unit, subsampling, center_offset)
                - x_offset
            )
            for ax in axes:
                ax.axvline(x, color="0.15", linestyle=":", linewidth=linewidth, alpha=0.2)


def _draw_top_piece_labels(
    ax,
    selected_segments,
    full_token_pieces,
    x_axis_unit: str,
    subsampling: int,
    center_offset: float,
    x_offset: float,
    fontsize: float,
    stagger: bool,
) -> None:
    base_y = 1.02
    row_gap = 0.12
    last_mid = None
    last_row = 0
    x_left, x_right = ax.get_xlim()
    close_threshold = 0.075 * max(x_right - x_left, 1e-8)
    for token_idx, start, end in selected_segments:
        mid = (
            _posterior_frame_to_x(
                (start + end) / 2,
                x_axis_unit,
                subsampling,
                center_offset,
            )
            - x_offset
        )
        row = 0
        if stagger and last_mid is not None and abs(mid - last_mid) < close_threshold:
            row = 1 - last_row
        y = base_y + row_gap * row
        ax.text(
            mid,
            y,
            _piece_label(full_token_pieces[token_idx]),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )
        last_mid = mid
        last_row = row


def _draw_top_word_labels(
    ax,
    sp,
    full_text: str,
    word_start: int,
    word_end: int,
    selected_segments,
    x_axis_unit: str,
    subsampling: int,
    center_offset: float,
    x_offset: float,
    fontsize: float,
) -> None:
    words = full_text.split()
    for word_idx, w_start, w_end in _word_token_ranges(sp, words):
        if word_idx < word_start or word_idx >= word_end:
            continue
        word_segments = [
            (start, end)
            for token_idx, start, end in selected_segments
            if w_start <= token_idx < w_end
        ]
        if not word_segments:
            continue
        start = min(s for s, _ in word_segments)
        end = max(e for _, e in word_segments)
        mid = (
            _posterior_frame_to_x(
                (start + end) / 2,
                x_axis_unit,
                subsampling,
                center_offset,
            )
            - x_offset
        )
        ax.text(
            mid,
            1.02,
            words[word_idx],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )


def _build_model(args: argparse.Namespace, device: torch.device) -> Conformer:
    lexicon = Lexicon(args.lang_dir)
    num_classes = max(lexicon.tokens) + 1
    saved = _checkpoint_metadata(_checkpoint_path(args.exp_dir, args.epoch))
    params = get_params()
    params.update(saved)
    if not hasattr(params, "num_decoder_layers"):
        params.num_decoder_layers = 6

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
        exp_dir=args.exp_dir,
        epoch=args.epoch,
        avg=args.avg,
        use_averaged_model=args.use_averaged_model,
        device=device,
    )
    return model


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.cuts_path = Path(args.cuts_path)
    args.out_png = Path(args.out_png)

    os.chdir(SCRIPT_DIR)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.lang_dir / "bpe.model"))
    blank_id = sp.piece_to_id("<blk>")

    cut = _select_cut(args.cuts_path, args.cut_index, args.cut_id)
    ref_text = cut.supervisions[0].text
    full_text = ref_text if args.text is None else args.text
    full_token_ids = sp.encode(full_text, out_type=int)
    full_token_pieces = sp.encode(full_text, out_type=str)
    word_start, word_end, token_start, token_end, window_text = _select_token_window(
        sp,
        text=full_text,
        word_start=args.word_start,
        num_words=args.num_words,
    )
    token_ids = full_token_ids[token_start:token_end]
    token_pieces = full_token_pieces[token_start:token_end]

    model = _build_model(args, device=device)
    feats = cut.load_features()
    x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        log_probs, _, _ = model(x, supervision=None, warmup=1.0)
        logp = log_probs[0].detach().cpu()
        post = logp.exp()
        T = logp.size(0)

    pred = torch.argmax(post, dim=-1).tolist()
    hyp_text = sp.decode(ctc_collapse(pred, blank_id=blank_id))
    max_prob = post.max(dim=-1).values
    entropy = -(post * logp).sum(dim=-1)

    print("Cut:", cut.id)
    print("REF:", ref_text)
    print("HYP:", hyp_text)
    print("Window text:", window_text)
    print("Model: baseline CTC")
    print(
        "Posterior stats:",
        {
            "argmax_nonblank_ratio": round(
                float((post.argmax(dim=-1) != blank_id).float().mean().item()), 6
            ),
            "mean_blank_prob": round(float(post[:, blank_id].mean().item()), 6),
            "mean_nonblank_prob": round(
                float((1.0 - post[:, blank_id]).mean().item()), 6
            ),
            "peakiness": round(float(max_prob.mean().item()), 6),
            "entropy_mean": round(float(entropy.mean().item()), 6),
        },
    )

    _, all_segments = ctc_forced_align_viterbi(logp, full_token_ids, blank_id=blank_id)
    selected_segments = [
        (u, start, end)
        for u, start, end in all_segments
        if token_start <= u < token_end
    ]
    if args.plot_full_cut:
        frame_start = 0
        frame_end = T
    elif selected_segments:
        context_frames = int(round(args.context_sec / (0.01 * 4)))
        frame_start = max(0, min(start for _, start, _ in selected_segments) - context_frames)
        frame_end = min(T, max(end for _, _, end in selected_segments) + context_frames)
    else:
        frame_start = 0
        frame_end = T

    subsampling = 4
    _, x_label = _make_x_values(T, args.x_axis_unit, subsampling=subsampling)
    x_values = [
        _posterior_frame_to_x(
            frame=i,
            x_axis_unit=args.x_axis_unit,
            subsampling=subsampling,
            center_offset=args.subsampling_center_offset,
        )
        for i in range(T)
    ]
    x_offset = (
        _input_frame_to_x(frame_start * subsampling, args.x_axis_unit, subsampling)
        if args.relative_x_axis
        else 0.0
    )
    if args.x_axis_unit == "time":
        x_label = "Timestep (sec)"
    if args.relative_x_axis:
        x_values = [x - x_offset for x in x_values]
        if args.x_axis_unit == "frame":
            x_label = "Output frame index from window start"
        elif args.x_axis_unit == "input_frame":
            x_label = "Input frame index from window start (10ms)"
    plot_slice = slice(frame_start, frame_end)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    if args.out_pdf is not None:
        args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if args.include_spectrogram:
        fig, (ax_spec, ax_post) = plt.subplots(
            2,
            1,
            figsize=(args.fig_width, args.fig_height),
            sharex=True,
            gridspec_kw={
                "height_ratios": [args.spec_height_ratio, args.post_height_ratio],
                "hspace": 0.06,
            },
        )
        _plot_spectrogram(
            ax_spec,
            feats=np.asarray(feats),
            frame_start=frame_start,
            frame_end=frame_end,
            x_axis_unit=args.x_axis_unit,
            subsampling=subsampling,
            x_offset=x_offset,
        )
        if args.top_label_mode == "pieces":
            _draw_top_piece_labels(
                ax_spec,
                selected_segments,
                full_token_pieces,
                args.x_axis_unit,
                subsampling,
                args.subsampling_center_offset,
                x_offset,
                args.token_label_size,
                args.top_label_stagger,
            )
        elif args.top_label_mode == "words":
            _draw_top_word_labels(
                ax_spec,
                sp,
                full_text,
                word_start,
                word_end,
                selected_segments,
                args.x_axis_unit,
                subsampling,
                args.subsampling_center_offset,
                x_offset,
                args.token_label_size,
            )
    else:
        fig, ax_post = plt.subplots(figsize=(args.fig_width, args.fig_height))
        ax_spec = None

    if args.show_blank:
        ax_post.plot(
            x_values[plot_slice],
            post[plot_slice, blank_id].numpy(),
            label="<blk>",
            linestyle="--",
            linewidth=args.blank_line_width,
            color="0.35",
            alpha=0.9,
        )
    for idx, token_id in enumerate(token_ids):
        if token_id == blank_id:
            continue
        ax_post.plot(
            x_values[plot_slice],
            post[plot_slice, token_id].numpy(),
            label=_piece_label(token_pieces[idx]),
            linewidth=args.line_width,
            color=_token_color(idx),
        )

    if args.show_token_boundaries:
        guide_axes = [ax_post] if ax_spec is None else [ax_spec, ax_post]
        _draw_token_boundaries(
            guide_axes,
            selected_segments,
            args.x_axis_unit,
            subsampling,
            args.subsampling_center_offset,
            x_offset,
            args.boundary_line_width,
        )

    ax_post.set_xlabel(x_label, fontsize=args.axis_label_size)
    ax_post.set_ylabel("Probability", fontsize=args.axis_label_size)
    ax_post.tick_params(axis="both", labelsize=args.tick_label_size)
    if ax_spec is not None:
        ax_spec.tick_params(axis="x", labelsize=args.tick_label_size)
    if args.title:
        fig.suptitle(args.title, y=0.99, fontsize=args.axis_label_size)
    if args.show_legend and len(token_ids) <= 16:
        ax_post.legend(loc="upper right", fontsize=7, ncol=2)

    y_text = -0.08
    if args.label_mode == "pieces":
        for token_idx, start, end in selected_segments:
            mid = (
                _posterior_frame_to_x(
                    (start + end) / 2,
                    args.x_axis_unit,
                    subsampling,
                    args.subsampling_center_offset,
                )
                - x_offset
            )
            label = _piece_label(full_token_pieces[token_idx])
            ax_post.text(
                mid,
                y_text,
                label,
                ha="center",
                va="top",
                fontsize=args.bottom_label_size,
            )
    elif args.label_mode == "words":
        words = full_text.split()
        for word_idx, w_start, w_end in _word_token_ranges(sp, words):
            if word_idx < word_start or word_idx >= word_end:
                continue
            word_segments = [
                (start, end)
                for token_idx, start, end in selected_segments
                if w_start <= token_idx < w_end
            ]
            if not word_segments:
                continue
            start = min(s for s, _ in word_segments)
            end = max(e for _, e in word_segments)
            mid = (
                _posterior_frame_to_x(
                    (start + end) / 2,
                    args.x_axis_unit,
                    subsampling,
                    args.subsampling_center_offset,
                )
                - x_offset
            )
            ax_post.text(
                mid,
                y_text,
                words[word_idx],
                ha="center",
                va="top",
                fontsize=args.bottom_label_size,
            )

    if frame_end > frame_start:
        if args.relative_x_axis:
            x_right = (
                _input_frame_to_x(frame_end * subsampling, args.x_axis_unit, subsampling)
                - x_offset
            )
            ax_post.set_xlim(0.0, x_right)
        else:
            ax_post.set_xlim(x_values[frame_start], x_values[frame_end - 1])
    y_bottom = -max(float(args.y_bottom_pad), 0.0)
    if args.label_mode != "none":
        y_bottom = min(y_bottom, -0.18)
    ax_post.set_ylim(bottom=y_bottom, top=1.05)
    if args.label_mode == "none":
        fig.subplots_adjust(bottom=0.13, top=0.84 if args.include_spectrogram else 0.95)
    else:
        fig.subplots_adjust(bottom=0.24, top=0.84 if args.include_spectrogram else 0.95)
    fig.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
    if args.out_pdf is not None:
        fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", args.out_png)
    if args.out_pdf is not None:
        print("Saved:", args.out_pdf)


if __name__ == "__main__":
    main()
