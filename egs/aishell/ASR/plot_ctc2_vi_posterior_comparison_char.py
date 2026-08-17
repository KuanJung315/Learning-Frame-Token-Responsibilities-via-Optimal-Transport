#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR / "conformer_ctc2"
PROJECT_ROOT = SCRIPT_DIR.parents[2]
LIBRISPEECH_RECIPE_DIR = PROJECT_ROOT / "egs" / "librispeech" / "ASR"

if str(RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(RECIPE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LIBRISPEECH_RECIPE_DIR) not in sys.path:
    sys.path.append(str(LIBRISPEECH_RECIPE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
import numpy as np
import torch
import torch.nn.functional as F

from icefall.utils import str2bool
from plot_ctc2_posterior import _build_model as build_baseline_model
from plot_ctc2_posterior import _piece_label, _token_color
from plot_vi_posterior import (
    _build_model as build_vi_model,
    _select_cut,
    _select_token_window,
    ctc_collapse,
    ctc_forced_align_viterbi,
)
from varctc_v2_utils import build_gated_log_probs_v2, encoder_lens_from_mask

CJK_FONT_PATH = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
CJK_FONT_PROP = (
    font_manager.FontProperties(fname=str(CJK_FONT_PATH))
    if CJK_FONT_PATH.is_file()
    else None
)


class CharTokenizer:
    def __init__(self, tokens_path: Path) -> None:
        self.symbol_to_id: Dict[str, int] = {}
        self.id_to_symbol: Dict[int, str] = {}
        with open(tokens_path, encoding="utf-8") as f:
            for line in f:
                symbol, token_id = line.rstrip().rsplit(maxsplit=1)
                token_id = int(token_id)
                self.symbol_to_id[symbol] = token_id
                self.id_to_symbol[token_id] = symbol
        self.unk_id = self.symbol_to_id["<unk>"]

    def encode(self, text: str, out_type=int):
        chars = [c for c in text if not c.isspace()]
        if out_type is str:
            return chars
        return [self.symbol_to_id.get(c, self.unk_id) for c in chars]

    def decode(self, token_ids: Sequence[int]) -> str:
        ignored = {"<blk>", "<sos/eos>", "<unk>"}
        return "".join(
            self.id_to_symbol.get(int(token_id), "")
            for token_id in token_ids
            if self.id_to_symbol.get(int(token_id), "") not in ignored
        )

    def piece_to_id(self, symbol: str) -> int:
        return self.symbol_to_id[symbol]


def _select_char_window(
    tokenizer: CharTokenizer,
    text: str,
    char_start: int,
    num_chars: int,
) -> Tuple[int, int, int, int, str]:
    chars = tokenizer.encode(text, out_type=str)
    if not chars:
        raise ValueError("Cannot plot empty text.")
    char_start = max(0, min(int(char_start), len(chars) - 1))
    char_end = len(chars) if num_chars <= 0 else min(len(chars), char_start + num_chars)
    return char_start, char_end, char_start, char_end, "".join(chars[char_start:char_end])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--baseline-exp-dir",
        type=Path,
        default=Path("conformer_ctc2/exp_ctc2_char_ctc_only_small256"),
    )
    parser.add_argument("--baseline-epoch", type=int, default=30)
    parser.add_argument("--baseline-avg", type=int, default=10)
    parser.add_argument(
        "--vi-exp-dir",
        type=Path,
        default=Path(
            "conformer_ctc2/"
            "exp_vi_ot_v2_char_ctc_only_small256_floor018_lam0005_ot10_batched"
        ),
    )
    parser.add_argument("--vi-epoch", type=int, default=40)
    parser.add_argument("--vi-avg", type=int, default=10)
    parser.add_argument("--use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_char"))
    parser.add_argument(
        "--cuts-path",
        type=Path,
        default=Path("data/fbank/aishell_cuts_dev.jsonl.gz"),
    )
    parser.add_argument("--cut-index", type=int, default=0)
    parser.add_argument("--cut-id", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument(
        "--word-start",
        type=int,
        default=0,
        help="First Chinese character to show.",
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=6,
        help="Number of Chinese characters to show; 0 shows the whole utterance.",
    )
    parser.add_argument("--context-sec", type=float, default=0.12)
    parser.add_argument(
        "--gate-source",
        type=str,
        default="prior",
        choices=["prior", "posterior", "mix"],
    )
    parser.add_argument("--mix-alpha", type=float, default=0.5)
    parser.add_argument("--prior-logit-bias", type=float, default=0.0)
    parser.add_argument("--label-embed-dim", type=int, default=256)
    parser.add_argument("--init-blank-prob", type=float, default=0.35)
    parser.add_argument("--show-blank", type=str2bool, default=True)
    parser.add_argument("--show-panel-labels", type=str2bool, default=True)
    parser.add_argument("--show-top-bpe-labels", type=str2bool, default=False)
    parser.add_argument("--show-bottom-bpe-labels", type=str2bool, default=True)
    parser.add_argument(
        "--plot-spectrogram",
        type=str2bool,
        default=True,
        help="Show the spectrogram panel above the posterior comparison.",
    )
    parser.add_argument(
        "--top-label-source",
        type=str,
        default="average",
        choices=["baseline", "vi", "average"],
        help="Which forced-alignment positions to use for BPE labels above the spectrogram.",
    )
    parser.add_argument(
        "--top-label-stagger",
        type=str2bool,
        default=True,
        help="Stagger close BPE labels above the spectrogram to reduce overlap.",
    )
    parser.add_argument("--fig-width", type=float, default=8.2)
    parser.add_argument("--fig-height", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--line-width", type=float, default=1.65)
    parser.add_argument("--blank-line-width", type=float, default=1.5)
    parser.add_argument("--axis-label-size", type=float, default=10.0)
    parser.add_argument("--tick-label-size", type=float, default=9.0)
    parser.add_argument("--bottom-label-size", type=float, default=8.0)
    parser.add_argument("--panel-label-size", type=float, default=9.0)
    parser.add_argument("--spec-height-ratio", type=float, default=0.8)
    parser.add_argument("--post-height-ratio", type=float, default=1.25)
    parser.add_argument("--hspace", type=float, default=0.22)
    parser.add_argument("--subsampling", type=int, default=4)
    parser.add_argument(
        "--subsampling-center-offset",
        type=float,
        default=3.0,
        help="Input-frame center offset for each subsampled posterior frame.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("conformer_ctc2/posterior_viz/comparison.png"),
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Optional PDF output for paper inclusion.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def _posterior_frame_to_sec(
    frame: float,
    subsampling: int,
    center_offset: float,
    x_offset: float,
) -> float:
    return (frame * subsampling + center_offset) * 0.01 - x_offset


def _input_frame_to_sec(input_frame: float, x_offset: float) -> float:
    return input_frame * 0.01 - x_offset


def _selected_segments(
    segments: Sequence[Tuple[int, int, int]],
    token_start: int,
    token_end: int,
) -> List[Tuple[int, int, int]]:
    return [
        (u, start, end)
        for u, start, end in segments
        if token_start <= u < token_end
    ]


def _common_frame_window(
    baseline_segments: Sequence[Tuple[int, int, int]],
    vi_segments: Sequence[Tuple[int, int, int]],
    T: int,
    context_sec: float,
    subsampling: int,
) -> Tuple[int, int]:
    merged = list(baseline_segments) + list(vi_segments)
    if not merged:
        return 0, T
    context_frames = int(round(context_sec / (0.01 * subsampling)))
    frame_start = max(0, min(start for _, start, _ in merged) - context_frames)
    frame_end = min(T, max(end for _, _, end in merged) + context_frames)
    return frame_start, frame_end


def _plot_spectrogram(
    ax,
    feats: np.ndarray,
    frame_start: int,
    frame_end: int,
    subsampling: int,
    x_offset: float,
) -> None:
    input_start = max(0, frame_start * subsampling)
    input_end = min(feats.shape[0], max(input_start + 1, frame_end * subsampling))
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
            _input_frame_to_sec(input_start, x_offset),
            _input_frame_to_sec(input_end, x_offset),
            0,
            feats.shape[1],
        ],
    )
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def _segment_mid_sec(
    segment: Tuple[int, int, int],
    subsampling: int,
    center_offset: float,
    x_offset: float,
) -> float:
    _, start, end = segment
    return _posterior_frame_to_sec(
        (start + end) / 2,
        subsampling=subsampling,
        center_offset=center_offset,
        x_offset=x_offset,
    )


def _draw_top_bpe_labels(
    ax,
    baseline_segments: Sequence[Tuple[int, int, int]],
    vi_segments: Sequence[Tuple[int, int, int]],
    full_token_pieces: Sequence[str],
    x_offset: float,
    args: argparse.Namespace,
) -> None:
    baseline_by_token = {token_idx: seg for token_idx, *seg in baseline_segments}
    baseline_by_token = {
        token_idx: (token_idx, start, end)
        for token_idx, (start, end) in baseline_by_token.items()
    }
    vi_by_token = {token_idx: seg for token_idx, *seg in vi_segments}
    vi_by_token = {
        token_idx: (token_idx, start, end)
        for token_idx, (start, end) in vi_by_token.items()
    }

    token_indices = sorted(set(baseline_by_token) | set(vi_by_token))
    x_left, x_right = ax.get_xlim()
    close_threshold = 0.055 * max(x_right - x_left, 1e-8)
    last_mid: Optional[float] = None
    last_row = 0
    for token_idx in token_indices:
        base_seg = baseline_by_token.get(token_idx)
        vi_seg = vi_by_token.get(token_idx)
        if args.top_label_source == "baseline":
            seg = base_seg or vi_seg
            if seg is None:
                continue
            mid = _segment_mid_sec(
                seg,
                subsampling=args.subsampling,
                center_offset=args.subsampling_center_offset,
                x_offset=x_offset,
            )
        elif args.top_label_source == "vi":
            seg = vi_seg or base_seg
            if seg is None:
                continue
            mid = _segment_mid_sec(
                seg,
                subsampling=args.subsampling,
                center_offset=args.subsampling_center_offset,
                x_offset=x_offset,
            )
        else:
            mids = []
            for seg in (base_seg, vi_seg):
                if seg is not None:
                    mids.append(
                        _segment_mid_sec(
                            seg,
                            subsampling=args.subsampling,
                            center_offset=args.subsampling_center_offset,
                            x_offset=x_offset,
                        )
                    )
            if not mids:
                continue
            mid = sum(mids) / len(mids)

        row = 0
        if (
            args.top_label_stagger
            and last_mid is not None
            and abs(mid - last_mid) < close_threshold
        ):
            row = 1 - last_row
        ax.text(
            mid,
            1.03 + 0.13 * row,
            _piece_label(full_token_pieces[token_idx]),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=args.bottom_label_size,
            fontproperties=CJK_FONT_PROP,
            clip_on=False,
        )
        last_mid = mid
        last_row = row


def _plot_posteriors(
    ax,
    post: torch.Tensor,
    token_ids: Sequence[int],
    token_pieces: Sequence[str],
    full_token_pieces: Sequence[str],
    selected_segments: Sequence[Tuple[int, int, int]],
    blank_id: int,
    frame_start: int,
    frame_end: int,
    x_values: np.ndarray,
    x_offset: float,
    subsampling: int,
    center_offset: float,
    panel_label: str,
    args: argparse.Namespace,
) -> None:
    plot_slice = slice(frame_start, frame_end)
    if args.show_blank:
        ax.plot(
            x_values[plot_slice],
            post[plot_slice, blank_id].numpy(),
            label="<blk>",
            linestyle="--",
            linewidth=args.blank_line_width,
            color="tab:blue",
            alpha=0.95,
        )

    for idx, token_id in enumerate(token_ids):
        if token_id == blank_id:
            continue
        ax.plot(
            x_values[plot_slice],
            post[plot_slice, token_id].numpy(),
            label=_piece_label(token_pieces[idx]),
            linewidth=args.line_width,
            color=_token_color(idx),
        )

    if args.show_bottom_bpe_labels:
        for token_idx, start, end in selected_segments:
            mid = _posterior_frame_to_sec(
                (start + end) / 2,
                subsampling=subsampling,
                center_offset=center_offset,
                x_offset=x_offset,
            )
            ax.text(
                mid,
                -0.08,
                _piece_label(full_token_pieces[token_idx]),
                ha="center",
                va="top",
                fontsize=args.bottom_label_size,
            )

    y_bottom = -0.18 if args.show_bottom_bpe_labels else -0.03
    ax.set_ylim(bottom=y_bottom, top=1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel("Probability", fontsize=args.axis_label_size)
    ax.tick_params(axis="both", labelsize=args.tick_label_size)
    if args.show_panel_labels:
        ax.text(
            0.01,
            0.94,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=args.panel_label_size,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )


def _print_stats(name: str, post: torch.Tensor, logp: torch.Tensor, blank_id: int) -> None:
    max_prob = post.max(dim=-1).values
    entropy = -(post * logp).sum(dim=-1)
    print(
        name,
        {
            "argmax_nonblank_ratio": round(
                float((post.argmax(dim=-1) != blank_id).float().mean().item()), 6
            ),
            "mean_blank_prob": round(float(post[:, blank_id].mean().item()), 6),
            "mean_nonblank_prob": round(float((1.0 - post[:, blank_id]).mean().item()), 6),
            "peakiness": round(float(max_prob.mean().item()), 6),
            "entropy_mean": round(float(entropy.mean().item()), 6),
        },
    )


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    args.out_png = Path(args.out_png)
    args.out_pdf = Path(args.out_pdf) if args.out_pdf is not None else None

    os.chdir(SCRIPT_DIR)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if CJK_FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(CJK_FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(CJK_FONT_PATH)
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    tokenizer = CharTokenizer(args.lang_dir / "tokens.txt")
    blank_id = tokenizer.piece_to_id("<blk>")

    cut = _select_cut(args.cuts_path, args.cut_index, args.cut_id)
    ref_text = cut.supervisions[0].text
    full_text = ref_text if args.text is None else args.text
    full_token_ids = tokenizer.encode(full_text, out_type=int)
    full_token_pieces = tokenizer.encode(full_text, out_type=str)
    word_start, word_end, token_start, token_end, window_text = _select_char_window(
        tokenizer,
        text=full_text,
        char_start=args.word_start,
        num_chars=args.num_words,
    )
    token_ids = full_token_ids[token_start:token_end]
    token_pieces = full_token_pieces[token_start:token_end]

    feats = cut.load_features()
    x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)

    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.exp_dir = args.baseline_exp_dir
    baseline_args.epoch = args.baseline_epoch
    baseline_args.avg = args.baseline_avg
    vi_args = argparse.Namespace(**vars(args))
    vi_args.exp_dir = args.vi_exp_dir
    vi_args.epoch = args.vi_epoch
    vi_args.avg = args.vi_avg

    baseline_model = build_baseline_model(baseline_args, device=device)
    vi_model = build_vi_model(vi_args, device=device)

    with torch.no_grad():
        baseline_log_probs, _, _ = baseline_model(x, supervision=None, warmup=1.0)
        baseline_logp = baseline_log_probs[0].detach().cpu()

        targets = torch.tensor(full_token_ids, dtype=torch.long, device=device).unsqueeze(0)
        target_lengths = torch.tensor([len(full_token_ids)], dtype=torch.long, device=device)
        memory, memory_key_padding_mask = vi_model.encoder.run_encoder(
            x, supervisions=None, warmup=1.0
        )
        encoder_out = memory.permute(1, 0, 2)
        encoder_lens = encoder_lens_from_mask(
            memory_key_padding_mask,
            batch_size=encoder_out.size(0),
            max_len=encoder_out.size(1),
            device=encoder_out.device,
        )
        log_p_nonblank = F.log_softmax(vi_model.ctc_head(encoder_out), dim=-1)
        alpha_prior = vi_model.blank_prior(encoder_out, encoder_lens)
        alpha_for_plot = alpha_prior
        if args.gate_source in ("posterior", "mix"):
            alpha_post = vi_model.blank_gate(
                encoder_out,
                targets,
                target_lengths,
                encoder_lens,
            )
            if args.gate_source == "posterior":
                alpha_for_plot = alpha_post
            else:
                mix_alpha = min(max(float(args.mix_alpha), 0.0), 1.0)
                alpha_for_plot = (1.0 - mix_alpha) * alpha_prior + mix_alpha * alpha_post
        if args.prior_logit_bias != 0.0 and args.gate_source in ("prior", "mix"):
            alpha_for_plot = alpha_for_plot.float().clamp(1.0e-5, 1.0 - 1.0e-5)
            alpha_for_plot = torch.sigmoid(
                torch.logit(alpha_for_plot) + args.prior_logit_bias
            )
        vi_log_probs = build_gated_log_probs_v2(log_p_nonblank, alpha_for_plot)
        vi_T = int(encoder_lens[0].item())
        vi_logp = vi_log_probs[0, :vi_T].detach().cpu()

    T = min(baseline_logp.size(0), vi_logp.size(0))
    baseline_logp = baseline_logp[:T]
    vi_logp = vi_logp[:T]
    baseline_post = baseline_logp.exp()
    vi_post = vi_logp.exp()

    baseline_hyp = tokenizer.decode(
        ctc_collapse(torch.argmax(baseline_post, dim=-1).tolist(), blank_id=blank_id)
    )
    vi_hyp = tokenizer.decode(
        ctc_collapse(torch.argmax(vi_post, dim=-1).tolist(), blank_id=blank_id)
    )
    print("Cut:", cut.id)
    print("REF:", ref_text)
    print("Window text:", window_text)
    print("Word window:", {"start": word_start, "end": word_end})
    print("Baseline HYP:", baseline_hyp)
    print("VI/OT HYP:", vi_hyp)
    print("VI/OT gate source:", args.gate_source)
    print("VI/OT prior logit bias:", args.prior_logit_bias)
    _print_stats("Baseline stats:", baseline_post, baseline_logp, blank_id)
    _print_stats("VI/OT stats:", vi_post, vi_logp, blank_id)

    _, baseline_segments_all = ctc_forced_align_viterbi(
        baseline_logp, full_token_ids, blank_id=blank_id
    )
    _, vi_segments_all = ctc_forced_align_viterbi(
        vi_logp, full_token_ids, blank_id=blank_id
    )
    baseline_segments = _selected_segments(
        baseline_segments_all, token_start=token_start, token_end=token_end
    )
    vi_segments = _selected_segments(
        vi_segments_all, token_start=token_start, token_end=token_end
    )
    frame_start, frame_end = _common_frame_window(
        baseline_segments,
        vi_segments,
        T=T,
        context_sec=args.context_sec,
        subsampling=args.subsampling,
    )
    x_offset = frame_start * args.subsampling * 0.01
    x_values = np.asarray(
        [
            _posterior_frame_to_sec(
                i,
                subsampling=args.subsampling,
                center_offset=args.subsampling_center_offset,
                x_offset=x_offset,
            )
            for i in range(T)
        ]
    )
    x_right = frame_end * args.subsampling * 0.01 - x_offset

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    if args.out_pdf is not None:
        args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if args.plot_spectrogram:
        fig, (ax_spec, ax_base, ax_vi) = plt.subplots(
            3,
            1,
            figsize=(args.fig_width, args.fig_height),
            sharex=True,
            gridspec_kw={
                "height_ratios": [
                    args.spec_height_ratio,
                    args.post_height_ratio,
                    args.post_height_ratio,
                ],
                "hspace": args.hspace,
            },
        )
        _plot_spectrogram(
            ax_spec,
            feats=np.asarray(feats),
            frame_start=frame_start,
            frame_end=frame_end,
            subsampling=args.subsampling,
            x_offset=x_offset,
        )
        if args.show_top_bpe_labels:
            _draw_top_bpe_labels(
                ax_spec,
                baseline_segments=baseline_segments,
                vi_segments=vi_segments,
                full_token_pieces=full_token_pieces,
                x_offset=x_offset,
                args=args,
            )
    else:
        fig, (ax_base, ax_vi) = plt.subplots(
            2,
            1,
            figsize=(args.fig_width, args.fig_height),
            sharex=True,
            gridspec_kw={
                "height_ratios": [
                    args.post_height_ratio,
                    args.post_height_ratio,
                ],
                "hspace": args.hspace,
            },
        )
    _plot_posteriors(
        ax_base,
        post=baseline_post,
        token_ids=token_ids,
        token_pieces=token_pieces,
        full_token_pieces=full_token_pieces,
        selected_segments=baseline_segments,
        blank_id=blank_id,
        frame_start=frame_start,
        frame_end=frame_end,
        x_values=x_values,
        x_offset=x_offset,
        subsampling=args.subsampling,
        center_offset=args.subsampling_center_offset,
        panel_label="Baseline",
        args=args,
    )
    _plot_posteriors(
        ax_vi,
        post=vi_post,
        token_ids=token_ids,
        token_pieces=token_pieces,
        full_token_pieces=full_token_pieces,
        selected_segments=vi_segments,
        blank_id=blank_id,
        frame_start=frame_start,
        frame_end=frame_end,
        x_values=x_values,
        x_offset=x_offset,
        subsampling=args.subsampling,
        center_offset=args.subsampling_center_offset,
        panel_label="VI/OT",
        args=args,
    )
    ax_vi.set_xlabel("Timestep (sec)", fontsize=args.axis_label_size)
    ax_vi.set_xlim(0.0, x_right)
    ax_vi.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_vi.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if (not args.plot_spectrogram) and args.show_top_bpe_labels:
        _draw_top_bpe_labels(
            ax_base,
            baseline_segments=baseline_segments,
            vi_segments=vi_segments,
            full_token_pieces=full_token_pieces,
            x_offset=x_offset,
            args=args,
        )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.98, bottom=0.10)
    fig.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
    if args.out_pdf is not None:
        fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", args.out_png)
    if args.out_pdf is not None:
        print("Saved:", args.out_pdf)


if __name__ == "__main__":
    main()
