#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang,
#                                                  Mingshuang Luo,
#                                                  Zengwei Yao,
#                                                  Quandong Wang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Full VI training: gated CTC + blank KL + OT alignment (alpha row / BPE-length column).

Full ELBO:
  L = L_CTC_gated  +  lambda_kl * sum_t KL(Bern(alpha_t) || Bern(alpha_tilde_t))
                    +  lambda_ot * L_align_OT

Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./conformer_ctc2/train_vi_ot.py \\
  --world-size 4 \\
  --num-epochs 30 \\
  --start-epoch 1 \\
  --exp-dir conformer_ctc2/exp_vi_ot \\
  --full-libri 1 \\
  --att-rate 0.0 \\
  --num-decoder-layers 0 \\
  --lambda-ot 0.1 \\
  --lambda-kl-blank 0.01 \\
  --max-duration 300
"""

import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import k2
import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from blank_gate import BlankGateHead, BlankPriorHead
from conformer import Conformer
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from optim import Eden, Eve
from ot_prior import vi_ot_loss_from_logprobs
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    create_grad_scaler,
    encode_supervisions,
    setup_logger,
    str2bool,
    torch_autocast,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


# ── Model wrapper ──────────────────────────────────────────────────────────────


def _gate_mix(batch_idx: int, warmup_start: int, warmup_steps: int) -> float:
    """
    Linear ramp for posterior warm-up.

    Returns 0.0 before warmup_start, then linearly ramps to 1.0 over
    warmup_steps batches.  1.0 means full posterior; 0.0 means prior-only.
    """
    if warmup_steps <= 0:
        return 1.0
    if batch_idx <= warmup_start:
        return 0.0
    return min(1.0, (batch_idx - warmup_start) / warmup_steps)


class ConformerVI(nn.Module):
    """
    Conformer encoder + blank gate (posterior q_psi) + blank prior (p_beta).

    All three components are saved/loaded as one checkpoint via state_dict().
    At inference, use `blank_prior` in place of `blank_gate` so the model
    only requires X (no Y), eliminating the training→inference gap.
    """

    def __init__(
        self, conformer: Conformer, blank_gate: BlankGateHead, blank_prior: BlankPriorHead
    ) -> None:
        super().__init__()
        self.encoder = conformer
        self.blank_gate = blank_gate
        self.blank_prior = blank_prior

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device


# ── Per-frame helpers ──────────────────────────────────────────────────────────


def _gate_log_probs_batch(
    log_probs: Tensor,
    alpha: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """
    Batch-wise gated CTC log-prob computation.

    log_probs: [N, T, V]  standard CTC log-softmax (blank at column 0)
    alpha:     [N, T]     non-blank gate probabilities from BlankGateHead
                          (0.0 at padding positions)
    Returns:   [N, T, V]  gated log-softmax where:
        p_gated(blank | t)     = 1 - alpha_t
        p_gated(k != blank | t) = alpha_t * p_theta(k | t) / (1 - p_theta(blank | t))

    Numerically stable: uses logsumexp for the non-blank normaliser.
    """
    # Do gate algebra in fp32 even under autocast.  In fp16, eps values such as
    # 1e-8 round to 0/1 at the probability boundaries and can create NaNs.
    log_probs = log_probs.float()
    alpha = alpha.float().clamp(eps, 1.0 - eps)

    # log(sum_{k!=blank} p_theta(k | t))  — [N, T], logsumexp is stable
    log_nonblank_sum = torch.logsumexp(log_probs[:, :, 1:], dim=-1)

    # Padding frames have alpha=0 before clamping, so they become almost-blank
    # rather than exact-blank.  That keeps the log domain finite for k2/OT.
    log_1_minus_alpha = torch.log1p(-alpha)                       # [N, T]

    # log(alpha_t) + log p_theta(k) - log(sum_nonblank)
    log_alpha = torch.log(alpha)                                   # [N, T]
    log_p_gated_nonblank = (
        log_alpha.unsqueeze(-1)           # [N, T, 1]
        + log_probs[:, :, 1:]             # [N, T, V-1]
        - log_nonblank_sum.unsqueeze(-1)  # [N, T, 1]
    )  # [N, T, V-1]

    return torch.cat([log_1_minus_alpha.unsqueeze(-1), log_p_gated_nonblank], dim=-1)


def _blank_kl_loss(alpha: Tensor, alpha_prior: Tensor, eps: float = 1e-5) -> Tensor:
    """
    KL(Bern(alpha.detach()) || Bern(alpha_prior))  averaged over T frames.

    alpha:       [T]  blank gate posterior alpha_t      (BlankGateHead) — stop-grad
    alpha_prior: [T]  blank gate prior alpha_tilde_t    (BlankPriorHead) — has grad

    Stop-gradient on posterior: the prior is trained to chase the posterior
    (more informative), not vice versa.  Posterior gets gradient only from
    the gated CTC loss, not from KL — prevents KL from collapsing the posterior
    toward the less expressive prior.  (Same design as VarCTC, ECCV 2020.)

    Returns per-frame KL so lambda_kl is length-invariant.
    """
    p = alpha.detach().float().clamp(eps, 1.0 - eps)   # posterior — stop-grad
    q = alpha_prior.float().clamp(eps, 1.0 - eps)       # prior — has grad
    kl = p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())
    return kl.mean()  # per-frame: lambda_kl is now length-invariant


def _build_padded_targets(
    token_ids: List[List[int]],
    device: torch.device,
    pad_value: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Pad token_ids to a (N, U_max) tensor for batch-level posterior computation.
    Returns (targets, target_lengths).
    """
    target_lengths = torch.tensor(
        [len(ids) for ids in token_ids], dtype=torch.long, device=device
    )
    max_len = int(target_lengths.max().item()) if target_lengths.numel() > 0 else 0
    targets = torch.full(
        (len(token_ids), max_len), fill_value=pad_value, dtype=torch.long, device=device
    )
    for i, ids in enumerate(token_ids):
        if ids:
            targets[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return targets, target_lengths


def _compute_bpe_lengths(token_ids: list, sp, device: torch.device) -> Tensor:
    """
    Float32 tensor of character counts for each BPE token.
    Strips the SentencePiece word-boundary marker (▁ / Ġ) before counting.
    Minimum length is 1 to avoid zero-mass tokens.
    """
    lengths = []
    for tid in token_ids:
        piece = sp.id_to_piece(tid)
        char_len = len(piece.replace("▁", "").replace("ġ", ""))
        lengths.append(max(1, char_len))
    return torch.tensor(lengths, dtype=torch.float32, device=device)


# ── Argument parsing ───────────────────────────────────────────────────────────


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=12354)
    parser.add_argument(
        "--tensorboard", type=str2bool, default=True,
        help="Log metrics to tensorboard."
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument(
        "--start-epoch", type=int, default=1,
        help="Resume from this epoch (loads epoch-{start_epoch-1}.pt)."
    )
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument(
        "--exp-dir", type=str, default="conformer_ctc2/exp_vi_ot",
        help="Experiment directory for checkpoints and logs."
    )
    parser.add_argument("--lang-dir", type=str, default="data/lang_bpe_500")
    parser.add_argument("--initial-lr", type=float, default=0.003)
    parser.add_argument("--lr-batches", type=float, default=5000)
    parser.add_argument("--lr-epochs", type=float, default=6)
    parser.add_argument(
        "--att-rate", type=float, default=0.0,
        help="Attention decoder rate. Set 0.0 for pure CTC+VI."
    )
    parser.add_argument("--num-decoder-layers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-diagnostics", type=str2bool, default=False)
    parser.add_argument("--save-every-n", type=int, default=8000)
    parser.add_argument("--keep-last-k", type=int, default=20)
    parser.add_argument("--average-period", type=int, default=100)
    parser.add_argument("--use-fp16", type=str2bool, default=False)
    # --full-libri and --max-duration are provided by LibriSpeechAsrDataModule.add_arguments()

    # ── OT arguments ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--lambda-ot", type=float, default=0.1,
        help="Weight for the OT alignment loss (L_align)."
    )
    parser.add_argument(
        "--ot-eps", type=float, default=0.5,
        help="Sinkhorn entropy regularization."
    )
    parser.add_argument(
        "--ot-iters", type=int, default=30,
        help="Number of Sinkhorn iterations."
    )
    parser.add_argument(
        "--ot-beta-pos", type=float, default=1.0,
        help="Diagonal positional bias for OT cost."
    )

    # ── VI / blank gate arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--lambda-kl-blank", type=float, default=0.01,
        help="Weight for blank posterior-prior KL (L_KL^blank)."
    )
    parser.add_argument(
        "--alpha-smooth-mix", type=float, default=0.1,
        help=(
            "Uniform mix weight for OT row marginal smoothing. "
            "Prevents a sharply peaked blank gate from making OT trivial."
        ),
    )
    parser.add_argument(
        "--col-marginal-type", type=str, default="bpe",
        choices=["bpe", "uniform"],
        help=(
            "'bpe': column marginal proportional to BPE token character length "
            "(theta-independent linguistic prior). "
            "'uniform': b_u = 1/U."
        ),
    )
    parser.add_argument(
        "--bpe-col-floor", type=float, default=0.05,
        help="Uniform floor for BPE-length column marginal (keeps all tokens reachable)."
    )

    # ── Blank gate architecture ────────────────────────────────────────────────
    parser.add_argument(
        "--label-embed-dim", type=int, default=256,
        help="Dimension for posterior label mean-pool embedding (Hadamard design).",
    )
    parser.add_argument(
        "--init-blank-prob", type=float, default=0.5,
        help=(
            "Initial non-blank probability for both prior and posterior gates "
            "(sigmoid(bias) at init). 0.5 = neutral; lower values bias toward blank."
        ),
    )

    # ── Posterior warm-up arguments ───────────────────────────────────────────
    parser.add_argument(
        "--gate-warmup-start", type=int, default=5000,
        help=(
            "Batch index at which to start mixing the posterior (cross-attn) gate. "
            "Before this, alpha_eff = alpha_prior only (stable prior-only phase)."
        ),
    )
    parser.add_argument(
        "--gate-warmup-steps", type=int, default=10000,
        help=(
            "Number of batches over which gate_mix linearly ramps from 0 to 1. "
            "alpha_eff = (1 - gate_mix) * alpha_prior + gate_mix * alpha_post."
        ),
    )

    # ── Standard conformer / CTC arguments ────────────────────────────────────
    parser.add_argument(
        "--beam-size", type=int, default=10,
        help="Beam size for k2.ctc_loss."
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,
            # Conformer architecture
            "feature_dim": 80,
            "subsampling_factor": 4,
            "encoder_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            # CTC
            "beam_size": 10,
            "reduction": "none",
            "use_double_scores": True,
            # Noam
            "model_warm_step": 3000,
            "env_info": get_env_info(),
        }
    )
    return params


# ── Checkpoint helpers ─────────────────────────────────────────────────────────


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler=None,
    rank: int = 0,
) -> None:
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )
    if params.best_train_epoch == params.cur_epoch:
        copyfile(src=filename, dst=params.exp_dir / "best-train-loss.pt")
    if params.best_valid_epoch == params.cur_epoch:
        copyfile(src=filename, dst=params.exp_dir / "best-valid-loss.pt")


# ── Core training loss ─────────────────────────────────────────────────────────


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    graph_compiler: Union[BpeCtcTrainingGraphCompiler, CtcTrainingGraphCompiler],
    is_training: bool,
    warmup: float = 1.0,
    debug: bool = False,
) -> Tuple[Tensor, MetricsTracker, Optional[Dict[str, Any]]]:
    """
    Full VI loss:
      L = L_CTC_gated  +  lambda_kl * L_KL^blank  +  lambda_ot * L_align
    """
    model_ref = model.module if hasattr(model, "module") else model
    device = next(model_ref.parameters()).device
    feature = batch["inputs"].to(device)
    assert feature.ndim == 3

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    with torch.set_grad_enabled(is_training):
        # nnet_output: [N, T', V] log-softmax
        # encoder_memory: [T', N, d_model]  (time-first — Transformer convention)
        # memory_mask:    [N, T']
        nnet_output, encoder_memory, memory_mask = model(
            feature, supervisions, warmup=warmup
        )
        # Permute to batch-first for blank gates; keep encoder_memory for AED decoder
        encoder_hidden = encoder_memory.permute(1, 0, 2)  # [N, T', d_model]

    encoder_out_lens = ((feature_lens - 1) // 2 - 1) // 2   # subsampling factor 4
    T_out = nnet_output.size(1)
    frame_idx = torch.arange(T_out, device=device).unsqueeze(0)
    valid_mask = frame_idx < encoder_out_lens.unsqueeze(1)

    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
        token_ids = graph_compiler.texts_to_ids(texts)
        decoding_graph = graph_compiler.compile(token_ids)
    elif isinstance(graph_compiler, CtcTrainingGraphCompiler):
        decoding_graph = graph_compiler.compile(texts)
        token_ids = None
    else:
        raise ValueError(f"Unsupported graph compiler type: {type(graph_compiler)}")

    # Posterior warm-up mix: ramp from 0 (prior-only) → 1 (full posterior)
    mix = _gate_mix(
        batch_idx=params.batch_idx_train,
        warmup_start=params.gate_warmup_start,
        warmup_steps=params.gate_warmup_steps,
    )

    # ── Blank gates (batch-level) ──────────────────────────────────────────────
    mmodel = model_ref

    with torch.set_grad_enabled(is_training):
        # Prior: batch-level (only X, no Y)
        alpha_prior_batch = mmodel.blank_prior(encoder_hidden, encoder_out_lens)  # [N, T']

    if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
        # Posterior: batch-level Hadamard (X + mean-pooled Y)
        targets_padded, target_lengths_t = _build_padded_targets(token_ids, device)
        sorted_to_orig = supervision_segments[:, 0].tolist()
        # supervision_segments is sorted; targets_padded rows are in sorted order.
        # Reorder to match encoder_hidden's original batch order.
        orig_order = torch.zeros(
            len(token_ids), dtype=torch.long, device=device
        )
        for sorted_idx, orig_idx in enumerate(sorted_to_orig):
            orig_order[orig_idx] = sorted_idx
        targets_orig = targets_padded[orig_order]
        tlen_orig = target_lengths_t[orig_order]

        with torch.set_grad_enabled(is_training):
            alpha_post_batch = mmodel.blank_gate(encoder_hidden, targets_orig, tlen_orig)
    else:
        alpha_post_batch = torch.zeros_like(alpha_prior_batch)

    # alpha_eff = (1-mix)*prior + mix*posterior.
    # At mix=0 this is prior-only: prior trains directly via gated CTC loss
    # (same design as VarCTC blank_source="prior" during warmup).
    # The 0.0*alpha_post term keeps blank_gate in the autograd graph so DDP
    # backward hooks fire even when mix=0 (find_unused_parameters=False).
    with torch.set_grad_enabled(is_training):
        alpha_eff_batch = (1.0 - mix) * alpha_prior_batch + mix * alpha_post_batch

    # ── Gated log-probs for CTC ────────────────────────────────────────────────
    with torch.set_grad_enabled(is_training):
        nnet_output_gated = _gate_log_probs_batch(nnet_output, alpha_eff_batch)  # [N, T', V]

    # ── CTC loss (gated emission) ──────────────────────────────────────────────

    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output_gated,
        supervision_segments,
        allow_truncate=params.subsampling_factor - 1,
    )
    ctc_loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    )
    ctc_loss_is_finite = torch.isfinite(ctc_loss)

    # ── Per-utterance KL and OT losses ────────────────────────────────────────
    lambda_ot = params.lambda_ot
    lambda_kl = params.lambda_kl_blank
    kl_losses: List[Tensor] = []
    ot_losses: List[Tensor] = []
    debug_info: Optional[Dict[str, Any]] = None

    if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
        for sorted_idx, ids in enumerate(token_ids):
            orig_idx = sorted_to_orig[sorted_idx]
            L = int(encoder_out_lens[orig_idx].item())

            prior_i     = alpha_prior_batch[orig_idx, :L]    # [T]
            post_i      = alpha_post_batch[orig_idx, :L]     # [T]
            alpha_eff_i = alpha_eff_batch[orig_idx, :L]      # [T]

            # KL(posterior.detach() || prior): only prior trains via KL.
            # Posterior trains only via gated CTC — prevents KL from pushing
            # the more informative posterior toward the simpler prior.
            kl_i = prior_i.new_tensor(0.0)
            if is_training and mix > 0.0:
                kl_i = _blank_kl_loss(post_i, prior_i)
            kl_losses.append(kl_i)

            # OT alignment loss — uses effective (mixed) alpha for row marginal
            if lambda_ot > 0:
                lp_gated = nnet_output_gated[orig_idx, :L]  # [T, V] gated log-probs
                labels = torch.tensor(ids, device=device, dtype=torch.long)

                if params.col_marginal_type == "bpe":
                    bpe_lengths = _compute_bpe_lengths(ids, graph_compiler.sp, device)
                else:
                    U = len(ids)
                    bpe_lengths = torch.ones(U, dtype=torch.float32, device=device)

                if debug and sorted_idx == 0:
                    ot_i, P, _ = vi_ot_loss_from_logprobs(
                        log_probs=lp_gated,
                        alpha=alpha_eff_i,
                        labels=labels,
                        bpe_lengths=bpe_lengths,
                        alpha_smooth_mix=params.alpha_smooth_mix,
                        bpe_col_floor=params.bpe_col_floor,
                        eps=params.ot_eps,
                        iters=params.ot_iters,
                        beta_pos=params.ot_beta_pos,
                        return_plan=True,
                    )
                    if P is not None:
                        token_pieces = [graph_compiler.sp.id_to_piece(t) for t in ids]
                        cuts = supervisions.get("cut", None)
                        cut_id = None
                        if cuts is not None and orig_idx < len(cuts):
                            cut = cuts[orig_idx]
                            cut_id = getattr(cut, "id", None)
                        debug_info = {"P": P, "token_pieces": token_pieces, "cut_id": cut_id}
                else:
                    ot_i = vi_ot_loss_from_logprobs(
                        log_probs=lp_gated,
                        alpha=alpha_eff_i,
                        labels=labels,
                        bpe_lengths=bpe_lengths,
                        alpha_smooth_mix=params.alpha_smooth_mix,
                        bpe_col_floor=params.bpe_col_floor,
                        eps=params.ot_eps,
                        iters=params.ot_iters,
                        beta_pos=params.ot_beta_pos,
                    )
                ot_losses.append(ot_i)
            else:
                ot_losses.append(alpha_eff_i.new_tensor(0.0))
    else:
        for _ in range(nnet_output.size(0)):
            kl_losses.append(nnet_output.new_tensor(0.0))
            ot_losses.append(nnet_output.new_tensor(0.0))

    kl_loss = torch.stack(kl_losses)
    ot_loss = torch.stack(ot_losses)

    if not torch.all(ctc_loss_is_finite):
        logging.info("Not all CTC losses are finite!")
        ctc_loss = ctc_loss[ctc_loss_is_finite]
        kl_loss = kl_loss[ctc_loss_is_finite]
        ot_loss = ot_loss[ctc_loss_is_finite]
        if torch.all(~ctc_loss_is_finite):
            raise ValueError("All losses are inf/nan — reduce max-duration.")

    # ── Total loss ─────────────────────────────────────────────────────────────
    if params.att_rate != 0.0:
        with torch.set_grad_enabled(is_training):
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = model_ref.encoder.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        loss = (
            (1.0 - params.att_rate) * ctc_loss.sum()
            + params.att_rate * att_loss
            + lambda_kl * kl_loss.sum()
            + lambda_ot * ot_loss.sum()
        )
    else:
        loss = (
            ctc_loss.sum()
            + lambda_kl * kl_loss.sum()
            + lambda_ot * ot_loss.sum()
        )
        att_loss = torch.tensor([0])

    # alpha_eff_batch = (1-mix)*prior + mix*post keeps blank_gate in the autograd
    # graph at all times (even when mix=0, the 0*post term creates a grad path),
    # so DDP backward hooks fire without a separate dummy term.

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
    valid_frames = valid_mask.sum().clamp_min(1).float()
    metric_frames = info["frames"]
    info["ctc_loss"] = ctc_loss.sum().detach().cpu().item()
    info["kl_blank_loss"] = kl_loss.sum().detach().cpu().item()
    info["ot_loss"] = ot_loss.sum().detach().cpu().item()
    info["gate_mix"] = mix * metric_frames
    info["alpha_prior_mean"] = (
        (alpha_prior_batch * valid_mask.to(alpha_prior_batch.dtype)).sum()
        / valid_frames
    ).detach().cpu().item() * metric_frames
    info["alpha_post_mean"] = (
        (alpha_post_batch * valid_mask.to(alpha_post_batch.dtype)).sum()
        / valid_frames
    ).detach().cpu().item() * metric_frames
    info["alpha_eff_mean"] = (
        (alpha_eff_batch * valid_mask.to(alpha_eff_batch.dtype)).sum()
        / valid_frames
    ).detach().cpu().item() * metric_frames
    if params.att_rate != 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()
    info["loss"] = loss.detach().cpu().item()
    info["utterances"] = feature.size(0)
    info["utt_duration"] = feature_lens.sum().item()
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    return loss, info, debug_info


# ── Validation ─────────────────────────────────────────────────────────────────


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: Union[BpeCtcTrainingGraphCompiler, CtcTrainingGraphCompiler],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    model.eval()
    tot_loss = MetricsTracker()
    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info, _ = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


# ── Training loop ──────────────────────────────────────────────────────────────


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    graph_compiler: Union[BpeCtcTrainingGraphCompiler, CtcTrainingGraphCompiler],
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: "GradScaler",
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    model.train()
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        with torch_autocast(enabled=params.use_fp16):
            loss, loss_info, debug_info = compute_loss(
                params=params,
                model=model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=True,
                warmup=min(1.0, params.batch_idx_train / params.model_warm_step),
                debug=False,
            )

        scheduler.step_batch(params.batch_idx_train)

        if params.use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if params.batch_idx_train % params.average_period == 0 and model_avg is not None:
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if params.batch_idx_train % params.save_every_n == 0:
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar("train/learning_rate", cur_lr, params.batch_idx_train)
                tb_writer.add_scalar(
                    "train/gate_mix",
                    _gate_mix(params.batch_idx_train, params.gate_warmup_start, params.gate_warmup_steps),
                    params.batch_idx_train,
                )
                loss_info.write_summary(tb_writer, "train/current_", params.batch_idx_train)
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            if tb_writer is not None:
                valid_info.write_summary(tb_writer, "train/valid_", params.batch_idx_train)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


# ── Main entry point ───────────────────────────────────────────────────────────


def run(rank: int, world_size: int, args) -> None:
    params = get_params()
    params.update(vars(args))
    if params.full_libri is False:
        params.valid_interval = 1600

    fix_random_seed(params.seed)
    dist_initialized = False
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)
        dist_initialized = True

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    if "lang_bpe" in str(params.lang_dir):
        graph_compiler = BpeCtcTrainingGraphCompiler(
            params.lang_dir,
            device=device,
            sos_token="<sos/eos>",
            eos_token="<sos/eos>",
        )
    elif "lang_phone" in str(params.lang_dir):
        assert params.att_rate == 0
        assert params.num_decoder_layers == 0
        graph_compiler = CtcTrainingGraphCompiler(lexicon, device=device)
        graph_compiler.sos_id = 1
        graph_compiler.eos_id = 1
    else:
        raise ValueError(f"Unsupported lang_dir: {params.lang_dir}")

    logging.info("Building ConformerVI model")
    conformer = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )
    blank_gate = BlankGateHead(
        d_model=params.encoder_dim,
        vocab_size=num_classes,
        label_embed_dim=params.label_embed_dim,
        init_blank_prob=params.init_blank_prob,
    )
    blank_prior = BlankPriorHead(
        d_model=params.encoder_dim,
        init_blank_prob=params.init_blank_prob,
    )
    model = ConformerVI(conformer, blank_gate, blank_prior)
    print(model)

    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        model_avg = copy.deepcopy(model)

    assert params.start_epoch > 0
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank])

    optimizer = Eve(model.parameters(), lr=params.initial_lr)
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        diagnostic = diagnostics.attach_diagnostics(model)

    librispeech = LibriSpeechAsrDataModule(args)

    if params.full_libri:
        train_cuts = librispeech.train_all_shuf_cuts()
    else:
        train_cuts = librispeech.train_clean_100_cuts()

    def remove_short_and_long_utt(c: Cut):
        return 1.0 <= c.duration <= 20.0

    def remove_invalid_utt_ctc(c: Cut):
        num_tokens = len(graph_compiler.texts_to_ids(c.supervisions[0].text))
        min_ratio = 0.0005
        max_ratio = 0.1
        return (
            num_tokens >= 2
            and num_tokens <= c.duration / params.subsampling_factor / params.feature_dim * 1000 * max_ratio
            and num_tokens >= c.duration * min_ratio
        )

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    train_dl = librispeech.train_dataloaders(train_cuts)

    valid_cuts = librispeech.dev_clean_cuts()
    valid_cuts += librispeech.dev_other_cuts()
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    scaler = create_grad_scaler(enabled=params.use_fp16)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    try:
        for epoch in range(params.start_epoch, params.num_epochs + 1):
            scheduler.step_epoch(epoch - 1)
            fix_random_seed(params.seed + epoch - 1)
            train_dl.sampler.set_epoch(epoch - 1)

            if tb_writer is not None:
                tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

            params.cur_epoch = epoch

            train_one_epoch(
                params=params,
                model=model,
                model_avg=model_avg,
                optimizer=optimizer,
                graph_compiler=graph_compiler,
                scheduler=scheduler,
                train_dl=train_dl,
                valid_dl=valid_dl,
                scaler=scaler,
                tb_writer=tb_writer,
                world_size=world_size,
                rank=rank,
            )

            if params.print_diagnostics:
                diagnostic.print_diagnostics()
                break

            save_checkpoint(
                params=params,
                model=model,
                model_avg=model_avg,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )

        logging.info("Done!")
    except Exception:
        logging.exception("Unhandled exception on rank %s", rank)
        raise
    finally:
        if tb_writer is not None:
            tb_writer.close()
        if dist_initialized:
            try:
                torch.distributed.barrier()
            finally:
                cleanup_dist()


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
