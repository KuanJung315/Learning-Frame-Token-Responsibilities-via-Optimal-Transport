#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lhotse import CutSet

from blank_gate_v2 import BlankGateHeadV2, BlankPriorHeadV2
from conformer import Conformer
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import str2bool
from train import get_params
from varctc_v2_utils import build_gated_log_probs_v2, encoder_lens_from_mask


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConformerVIV2ForPlot(nn.Module):
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path(
            "conformer_ctc2/exp_vi_ot_v2_aed_li100h_ot01_mix05_acousticcol_init035_eff035"
        ),
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
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text whose BPE token posteriors should be plotted. Default: cut reference.",
    )
    parser.add_argument(
        "--word-start",
        type=int,
        default=0,
        help="First word index to show. Alignment is still computed on the full text.",
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=8,
        help="Number of words to show. Use 0 to show the whole utterance.",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=0.25,
        help="Extra audio context on both sides of the selected word window.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="words",
        choices=["words", "pieces"],
        help="Use word labels on the x axis, or every BPE piece label.",
    )
    parser.add_argument(
        "--gate-source",
        type=str,
        default="prior",
        choices=["prior", "posterior", "mix"],
        help=(
            "'prior' is the actual inference/decode gate. 'posterior' uses the "
            "reference tokens and is for debugging only. 'mix' interpolates them."
        ),
    )
    parser.add_argument("--mix-alpha", type=float, default=0.5)
    parser.add_argument("--prior-logit-bias", type=float, default=0.0)
    parser.add_argument("--label-embed-dim", type=int, default=256)
    parser.add_argument("--init-blank-prob", type=float, default=0.35)
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
        "--out-png",
        type=Path,
        default=Path("conformer_ctc2/posterior_viz/vi_posterior.png"),
    )
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--fig-height", type=float, default=3.6)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show-title", type=str2bool, default=True)
    parser.add_argument("--show-legend", type=str2bool, default=True)
    parser.add_argument("--line-width", type=float, default=1.8)
    parser.add_argument("--blank-line-width", type=float, default=1.8)
    parser.add_argument("--axis-label-size", type=float, default=11.0)
    parser.add_argument("--tick-label-size", type=float, default=10.0)
    parser.add_argument("--bottom-label-size", type=float, default=9.0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def _piece_label(piece: str) -> str:
    return piece.replace("▁", "_")


def ctc_collapse(ids: List[int], blank_id: int = 0) -> List[int]:
    out = []
    prev = None
    for i in ids:
        if i == blank_id:
            prev = i
            continue
        if i != prev:
            out.append(i)
        prev = i
    return out


def ctc_forced_align_viterbi(
    logp: torch.Tensor,
    targets: List[int],
    blank_id: int = 0,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    logp = logp.detach().cpu()
    targets_t = torch.tensor(targets, dtype=torch.long)

    T = logp.size(0)
    U = targets_t.numel()
    if T < 2 * U + 1:
        raise ValueError(f"T={T} is too short for target length U={U}")

    ext = torch.full((2 * U + 1,), blank_id, dtype=torch.long)
    if U > 0:
        ext[1::2] = targets_t
    S = ext.numel()

    neg_inf = -1.0e9
    dp = torch.full((T, S), neg_inf)
    bp = torch.full((T, S), -1, dtype=torch.long)
    dp[0, 0] = logp[0, ext[0]]
    bp[0, 0] = 0
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
        bp[0, 1] = 1

    for t in range(1, T):
        for s in range(S):
            best_prev = s
            best_score = dp[t - 1, s]
            if s - 1 >= 0 and dp[t - 1, s - 1] > best_score:
                best_score = dp[t - 1, s - 1]
                best_prev = s - 1
            if s - 2 >= 0 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                if dp[t - 1, s - 2] > best_score:
                    best_score = dp[t - 1, s - 2]
                    best_prev = s - 2
            dp[t, s] = best_score + logp[t, ext[s]]
            bp[t, s] = best_prev

    last_state = 0 if S == 1 else (S - 1 if dp[T - 1, S - 1] > dp[T - 1, S - 2] else S - 2)
    path_states = []
    s = last_state
    for t in range(T - 1, -1, -1):
        path_states.append(s)
        s = int(bp[t, s].item())
    path_states.reverse()

    path_states_t = torch.tensor(path_states, dtype=torch.long)
    path_labels = ext[path_states_t]

    segments = []
    for u in range(U):
        state_idx = 2 * u + 1
        positions = (path_states_t == state_idx).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        start = int(positions[0].item())
        end = int(positions[-1].item()) + 1
        segments.append((u, start, end))

    return path_labels, segments


def _checkpoint_metadata(checkpoint_path: Path) -> Dict:
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


def _checkpoint_path(exp_dir: Path, epoch: int) -> Path:
    path = exp_dir / f"epoch-{epoch}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
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
            raise ValueError(f"epoch - avg must be >= 1, got {epoch} - {avg}")
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


def _build_model(args: argparse.Namespace, device: torch.device) -> ConformerVIV2ForPlot:
    lexicon = Lexicon(args.lang_dir)
    num_classes = max(lexicon.tokens) + 1
    saved = _checkpoint_metadata(_checkpoint_path(args.exp_dir, args.epoch))
    params = get_params()
    params.update(saved)
    if not hasattr(params, "num_decoder_layers"):
        params.num_decoder_layers = 6
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

    model = ConformerVIV2ForPlot(
        encoder=encoder,
        ctc_head=nn.Linear(params.encoder_dim, num_classes - 1),
        blank_gate=BlankGateHeadV2(
            d_model=params.encoder_dim,
            vocab_size=num_classes,
            d_attn=params.label_embed_dim,
            init_blank_prob=params.init_blank_prob,
        ),
        blank_prior=BlankPriorHeadV2(
            d_model=params.encoder_dim,
            init_blank_prob=params.init_blank_prob,
        ),
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


def _select_cut(cuts_path: Path, cut_index: int, cut_id: Optional[str]):
    cuts = CutSet.from_file(cuts_path)
    if cut_id is not None:
        for cut in cuts:
            supervision_ids = [sup.id for sup in getattr(cut, "supervisions", [])]
            if cut.id == cut_id or cut_id in supervision_ids or cut.id.startswith(f"{cut_id}-"):
                return cut
        raise ValueError(f"Cut/supervision id {cut_id} was not found in {cuts_path}")

    for idx, cut in enumerate(cuts):
        if idx == cut_index:
            return cut
    raise ValueError(f"cut-index {cut_index} out of range for {cuts_path}")


def _make_x_values(T: int, x_axis_unit: str, subsampling: int):
    frame_shift = 0.01
    if x_axis_unit == "frame":
        return list(range(T)), "Output frame index"
    if x_axis_unit == "input_frame":
        return [i * subsampling for i in range(T)], "Input frame index (10ms)"
    if x_axis_unit == "time":
        return [i * frame_shift * subsampling for i in range(T)], "Time (sec)"
    raise ValueError(f"Unsupported x-axis unit: {x_axis_unit}")


def _word_token_ranges(sp, words: List[str]) -> List[Tuple[int, int, int]]:
    ranges = []
    for i in range(len(words)):
        prefix = " ".join(words[:i])
        upto = " ".join(words[: i + 1])
        start = len(sp.encode(prefix, out_type=int)) if prefix else 0
        end = len(sp.encode(upto, out_type=int))
        ranges.append((i, start, end))
    return ranges


def _select_token_window(
    sp,
    text: str,
    word_start: int,
    num_words: int,
) -> Tuple[int, int, int, int, str]:
    words = text.split()
    if not words:
        raise ValueError("Cannot plot an empty text.")

    word_start = max(0, min(int(word_start), len(words) - 1))
    if num_words <= 0:
        word_end = len(words)
    else:
        word_end = min(len(words), word_start + int(num_words))

    ranges = _word_token_ranges(sp, words)
    token_start = ranges[word_start][1]
    token_end = ranges[word_end - 1][2]
    window_text = " ".join(words[word_start:word_end])
    return word_start, word_end, token_start, token_end, window_text


def _frame_to_x(frame: float, x_axis_unit: str, subsampling: int) -> float:
    if x_axis_unit == "frame":
        return frame
    if x_axis_unit == "input_frame":
        return frame * subsampling
    return frame * 0.01 * subsampling


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
    targets = torch.tensor(full_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    target_lengths = torch.tensor([len(full_token_ids)], dtype=torch.long, device=device)

    with torch.no_grad():
        memory, memory_key_padding_mask = model.encoder.run_encoder(
            x, supervisions=None, warmup=1.0
        )
        encoder_out = memory.permute(1, 0, 2)
        encoder_lens = encoder_lens_from_mask(
            memory_key_padding_mask,
            batch_size=encoder_out.size(0),
            max_len=encoder_out.size(1),
            device=encoder_out.device,
        )
        log_p_nonblank = F.log_softmax(model.ctc_head(encoder_out), dim=-1)
        alpha_prior = model.blank_prior(encoder_out, encoder_lens)

        alpha_for_plot = alpha_prior
        if args.gate_source in ("posterior", "mix"):
            alpha_post = model.blank_gate(
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

        log_probs = build_gated_log_probs_v2(log_p_nonblank, alpha_for_plot)
        T = int(encoder_lens[0].item())
        logp = log_probs[0, :T].detach().cpu()
        post = logp.exp()

    pred = torch.argmax(post, dim=-1).tolist()
    hyp_ids = ctc_collapse(pred, blank_id=blank_id)
    hyp_text = sp.decode(hyp_ids)
    max_prob = post.max(dim=-1).values
    entropy = -(post * logp).sum(dim=-1)

    print("Cut:", cut.id)
    print("REF:", ref_text)
    print("HYP:", hyp_text)
    print("Full plot text:", full_text)
    print("Window text:", window_text)
    print("Word window:", {"start": word_start, "end": word_end})
    print("Gate source:", args.gate_source)
    print("Prior logit bias:", args.prior_logit_bias)
    print(
        "Posterior stats:",
        {
            "argmax_nonblank_ratio": round(float((post.argmax(dim=-1) != blank_id).float().mean().item()), 6),
            "mean_blank_prob": round(float(post[:, blank_id].mean().item()), 6),
            "mean_nonblank_prob": round(float((1.0 - post[:, blank_id]).mean().item()), 6),
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
    if selected_segments:
        context_frames = int(round(args.context_sec / (0.01 * 4)))
        frame_start = max(0, min(start for _, start, _ in selected_segments) - context_frames)
        frame_end = min(T, max(end for _, _, end in selected_segments) + context_frames)
    else:
        frame_start = 0
        frame_end = T

    x_values, x_label = _make_x_values(
        T,
        x_axis_unit=args.x_axis_unit,
        subsampling=int(getattr(get_params(), "subsampling_factor", 4)),
    )
    x_offset = _frame_to_x(frame_start, args.x_axis_unit, subsampling=4)
    if args.relative_x_axis:
        x_values = [x - x_offset for x in x_values]
        if args.x_axis_unit == "frame":
            x_label = "Output frame index from window start"
        elif args.x_axis_unit == "input_frame":
            x_label = "Input frame index from window start (10ms)"
        elif args.x_axis_unit == "time":
            x_label = "Timestep (sec)"
    elif args.x_axis_unit == "time":
        x_label = "Timestep (sec)"

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(args.fig_width, args.fig_height))
    plot_slice = slice(frame_start, frame_end)
    plt.plot(
        x_values[plot_slice],
        post[plot_slice, blank_id].numpy(),
        label="<blk>",
        linestyle="--",
        linewidth=args.blank_line_width,
    )

    for idx, token_id in enumerate(token_ids):
        if token_id == blank_id:
            continue
        plt.plot(
            x_values[plot_slice],
            post[plot_slice, token_id].numpy(),
            label=token_pieces[idx],
            linewidth=args.line_width,
        )

    plt.xlabel(x_label, fontsize=args.axis_label_size)
    plt.ylabel("Probability", fontsize=args.axis_label_size)
    plt.tick_params(axis="both", labelsize=args.tick_label_size)
    if args.show_title:
        plt.title(
            f"VI CTC token posteriors over time "
            f"({args.gate_source}, bias={args.prior_logit_bias:g})\n{window_text}"
        )

    if args.show_legend and len(token_ids) <= 12:
        plt.legend(loc="upper right", fontsize=8, ncol=2)

    y_text = -0.08
    subsampling = 4
    if args.label_mode == "pieces":
        for token_idx, start, end in selected_segments:
            mid = _frame_to_x((start + end) / 2, args.x_axis_unit, subsampling)
            if args.relative_x_axis:
                mid -= x_offset
            label = _piece_label(full_token_pieces[token_idx])
            plt.text(
                mid,
                y_text,
                label,
                ha="center",
                va="top",
                fontsize=args.bottom_label_size,
            )
    else:
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
            mid = _frame_to_x((start + end) / 2, args.x_axis_unit, subsampling)
            if args.relative_x_axis:
                mid -= x_offset
            plt.text(
                mid,
                y_text,
                words[word_idx],
                ha="center",
                va="top",
                fontsize=args.bottom_label_size,
            )

    if frame_end > frame_start:
        if args.relative_x_axis:
            plt.xlim(0.0, _frame_to_x(frame_end, args.x_axis_unit, subsampling) - x_offset)
        else:
            plt.xlim(x_values[frame_start], x_values[frame_end - 1])
    plt.ylim(bottom=-0.18)
    plt.subplots_adjust(bottom=0.28)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
    print("Saved:", args.out_png)


if __name__ == "__main__":
    main()
