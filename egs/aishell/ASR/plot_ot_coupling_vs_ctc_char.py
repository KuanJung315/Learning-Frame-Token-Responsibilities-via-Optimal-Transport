#!/usr/bin/env python3
"""Visualize, for one AISHELL utterance and one VFTA (VarCTC v2) model, the raw
CTC posterior versus the OT transport coupling P.

Both panels are [token x frame] heatmaps over the SAME frame axis:

  * CTC posterior   : gated_posterior[t, token_id[u]]  -- the probability mass
                      the model assigns to each transcript char per frame
                      (spiky, near one-frame-per-token).
  * OT coupling P   : the Sinkhorn transport plan returned by
                      vi_ot_loss_v2(..., return_plan=True) (a structured, soft,
                      near-monotonic alignment).

The point of the figure: the OT term induces an alignment structure that the
raw CTC posterior does not have. It uses ONLY this model -- no other method is
run; this is an internal-mechanism visualization, not a baseline comparison.

The OT parameters MUST match training (col-marginal acoustic, eps 0.3, iters
10, beta_pos 1.0, the token-prior knobs) so the plotted plan is the real
training-time coupling.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR / "conformer_ctc2"
PROJECT_ROOT = SCRIPT_DIR.parents[2]
LIBRISPEECH_RECIPE_DIR = PROJECT_ROOT / "egs" / "librispeech" / "ASR"
for _p in (RECIPE_DIR, SCRIPT_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(LIBRISPEECH_RECIPE_DIR) not in sys.path:
    sys.path.append(str(LIBRISPEECH_RECIPE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn.functional as F

from icefall.utils import str2bool
from plot_vi_posterior import _build_model as build_vi_model, _select_cut
from plot_ctc2_vi_posterior_comparison_char import CharTokenizer, _select_char_window
from varctc_v2_utils import build_gated_log_probs_v2, encoder_lens_from_mask
from ot_prior_v2 import vi_ot_loss_v2

CJK_FONT_PATH = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
CJK_FONT_PROP = (
    font_manager.FontProperties(fname=str(CJK_FONT_PATH))
    if CJK_FONT_PATH.is_file()
    else None
)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # model (kept arg names compatible with plot_vi_posterior._build_model)
    p.add_argument(
        "--exp-dir",
        type=Path,
        default=Path(
            "conformer_ctc2/exp_vi_ot_v2_char_ctc_only_small256_"
            "floor018_lam0005_ot10_batched"
        ),
    )
    p.add_argument("--epoch", type=int, default=60)
    p.add_argument("--avg", type=int, default=20)
    p.add_argument("--use-averaged-model", type=str2bool, default=True)
    p.add_argument("--lang-dir", type=Path, default=Path("data/lang_char"))
    p.add_argument("--num-decoder-layers", type=int, default=0)
    p.add_argument("--encoder-dim", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--num-encoder-layers", type=int, default=12)
    p.add_argument("--label-embed-dim", type=int, default=256)
    p.add_argument("--init-blank-prob", type=float, default=0.35)
    # data / window
    p.add_argument(
        "--cuts-path", type=Path, default=Path("data/fbank/aishell_cuts_dev.jsonl.gz")
    )
    p.add_argument("--cut-id", type=str, default="BAC009S0754W0258-1335")
    p.add_argument("--cut-index", type=int, default=0)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--char-start", type=int, default=0)
    p.add_argument(
        "--num-chars", type=int, default=0, help="0 = all chars (no windowing)."
    )
    p.add_argument(
        "--frame-context", type=int, default=3,
        help="Frames of context to keep around the windowed tokens' OT support.",
    )
    p.add_argument(
        "--support-threshold",
        type=float,
        default=0.35,
        help="Relative frame-mass threshold used only to crop a token window; keeps the main support and drops tiny remote OT tails.",
    )
    # which alpha feeds the OT row marginal (training uses the posterior gate)
    p.add_argument(
        "--gate-source", choices=["prior", "posterior"], default="posterior"
    )
    # OT params -- MUST match training
    p.add_argument("--col-marginal-type", type=str, default="acoustic")
    p.add_argument("--ot-eps", type=float, default=0.3)
    p.add_argument("--ot-iters", type=int, default=10)
    p.add_argument("--ot-beta-pos", type=float, default=1.0)
    p.add_argument("--ot-token-prior-sigma", type=float, default=0.15)
    p.add_argument("--ot-token-prior-score-temp", type=float, default=1.0)
    p.add_argument("--ot-token-prior-floor", type=float, default=0.05)
    p.add_argument("--alpha-smooth-mix", type=float, default=0.1)
    # output
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("conformer_ctc2/posterior_viz/ot_coupling_vs_ctc.png"),
    )
    p.add_argument("--subsampling", type=int, default=4)
    p.add_argument("--frame-shift", type=float, default=0.01)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Shared sequential colormap for direct structural comparison.",
    )
    p.add_argument(
        "--max-y-labels",
        type=int,
        default=8,
        help="Maximum number of token labels to show without hiding heatmap rows.",
    )
    p.add_argument(
        "--show-panel-titles",
        type=str2bool,
        default=False,
        help="Draw panel titles above the heatmaps. Disable for paper figures whose caption explains the two views.",
    )
    p.add_argument("--fig-width", type=float, default=7.0)
    p.add_argument("--fig-height", type=float, default=4.65)
    p.add_argument("--device", type=str, default="cuda")
    return p


def main() -> None:
    args = get_parser().parse_args()
    os.chdir(SCRIPT_DIR)
    device = (
        torch.device(args.device)
        if (args.device != "cuda" or torch.cuda.is_available())
        else torch.device("cpu")
    )

    tokenizer = CharTokenizer(args.lang_dir / "tokens.txt")
    cut = _select_cut(args.cuts_path, args.cut_index, args.cut_id)
    ref_text = cut.supervisions[0].text
    full_text = ref_text if args.text is None else args.text
    full_token_ids = tokenizer.encode(full_text, out_type=int)
    full_token_chars = tokenizer.encode(full_text, out_type=str)
    char_start, char_end, _, _, window_text = _select_char_window(
        tokenizer, full_text, args.char_start, args.num_chars
    )

    model = build_vi_model(args, device=device)
    feats = cut.load_features()
    x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    targets = torch.tensor(full_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    target_lengths = torch.tensor(
        [len(full_token_ids)], dtype=torch.long, device=device
    )

    with torch.no_grad():
        memory, mask = model.encoder.run_encoder(x, supervisions=None, warmup=1.0)
        encoder_out = memory.permute(1, 0, 2)
        encoder_lens = encoder_lens_from_mask(
            mask, batch_size=encoder_out.size(0),
            max_len=encoder_out.size(1), device=encoder_out.device,
        )
        log_p_nonblank = F.log_softmax(model.ctc_head(encoder_out), dim=-1)  # [1,T,V-1]
        alpha_prior = model.blank_prior(encoder_out, encoder_lens)           # [1,T]
        if args.gate_source == "posterior":
            alpha = model.blank_gate(encoder_out, targets, target_lengths, encoder_lens)
        else:
            alpha = alpha_prior

        T = int(encoder_lens[0].item())
        labels = torch.tensor(full_token_ids, dtype=torch.long, device=device)

        # CTC posterior (gated) restricted to the transcript tokens -> [T, U]
        gated = build_gated_log_probs_v2(log_p_nonblank, alpha_prior)[0, :T].exp()
        ctc_on_tokens = gated[:, labels].detach().cpu().numpy()          # [T, U]

        # OT transport coupling P -> [T, U] (must match training params)
        _, P = vi_ot_loss_v2(
            log_p_nonblank[0, :T],
            alpha[0, :T],
            labels=labels,
            column_marginal_type=args.col_marginal_type,
            alpha_smooth_mix=args.alpha_smooth_mix,
            token_prior_sigma=args.ot_token_prior_sigma,
            token_prior_score_temp=args.ot_token_prior_score_temp,
            token_prior_floor=args.ot_token_prior_floor,
            eps=args.ot_eps,
            iters=args.ot_iters,
            beta_pos=args.ot_beta_pos,
            return_plan=True,
        )
        ot_coupling = P.detach().cpu().numpy()                            # [T, U]

    # optional token window + frame crop around the window's OT support
    u0, u1 = char_start, char_end
    ctc_w = ctc_on_tokens[:, u0:u1]
    ot_w = ot_coupling[:, u0:u1]
    chars_w = full_token_chars[u0:u1]
    if args.num_chars > 0:
        col_mass = ot_w.sum(axis=1)
        nz = np.where(col_mass > col_mass.max() * args.support_threshold)[0]
        if nz.size:
            f0 = max(0, nz[0] - args.frame_context)
            f1 = min(T, nz[-1] + 1 + args.frame_context)
        else:
            f0, f1 = 0, T
    else:
        f0, f1 = 0, T
    ctc_w = ctc_w[f0:f1].T  # [U, F]
    ot_w = ot_w[f0:f1].T    # [U, F]

    sec_per_frame = args.frame_shift * args.subsampling
    extent = [f0 * sec_per_frame, f1 * sec_per_frame, len(chars_w) - 0.5, -0.5]

    # The figure is intended for a single IEEE column. It is generated at
    # roughly twice the final width, so these font sizes become 7--9 pt after
    # LaTeX scales it to \columnwidth.
    fig, axes = plt.subplots(
        2, 1, figsize=(args.fig_width, args.fig_height), constrained_layout=True
    )
    max_y_labels = max(1, args.max_y_labels)
    ytick_stride = max(1, (len(chars_w) + max_y_labels - 1) // max_y_labels)
    ytick_positions = np.arange(0, len(chars_w), ytick_stride)
    for ax, mat, title, colorbar_label in (
        (axes[0], ctc_w, "(a) CTC posterior  $p_t(c_u)$", "prob."),
        (axes[1], ot_w, "(b) OT transport plan  $A^{\\star}_{t,u}$", "mass"),
    ):
        im = ax.imshow(mat, aspect="auto", origin="upper", extent=extent,
                       interpolation="nearest", cmap=args.cmap)
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(
            [chars_w[i] for i in ytick_positions],
            fontproperties=CJK_FONT_PROP,
            fontsize=14,
        )
        if args.show_panel_titles:
            ax.set_title(title, fontsize=16)
        ax.tick_params(axis="x", labelsize=14)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        colorbar.ax.tick_params(labelsize=13)
        colorbar.ax.set_title(colorbar_label, fontsize=13, pad=4)
    axes[1].set_xlabel("Time (s)", fontsize=16)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi)
    fig.savefig(args.out_png.with_suffix(".pdf"))
    print(f"Cut: {cut.id}")
    print(f"REF: {ref_text}")
    print(f"Window: '{window_text}'  tokens[{u0}:{u1}]  frames[{f0}:{f1}] of {T}")
    print(f"Saved: {args.out_png}  and  {args.out_png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
