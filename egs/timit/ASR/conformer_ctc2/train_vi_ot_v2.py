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
VarCTC v2 training: cross-attention posterior gate + V-1 CTC head + OT alignment.

Full ELBO:
  L = L_CTC_gated  +  lambda_kl * sum_t KL(Bern(alpha_t) || Bern(alpha_tilde_t))
                    +  lambda_ot * L_align_OT

Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./conformer_ctc2/train_vi_ot_v2.py \\
  --world-size 4 \\
  --num-epochs 30 \\
  --start-epoch 1 \\
  --exp-dir conformer_ctc2/exp_vi_ot_v2 \\
  --lang-dir data/lang_bbpe_500 \\
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
from asr_datamodule import TimitAsrDataModule
from phone_graph_compiler import PhoneCtcTrainingGraphCompiler
from blank_gate_v2 import BlankGateHeadV2, BlankPriorHeadV2
from conformer import Conformer
from ctc_plan_consistency import ctc_token_occupancy_batched, temporal_plan_w1_loss
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from optim import Eden, Eve
from ot_fgw import vi_fgw_loss_v2, vi_fgw_loss_v2_batched
from ot_prior_v2 import vi_ot_loss_v2, vi_ot_loss_v2_batched
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from varctc_v2_utils import (
    build_gated_log_probs_v2,
    encoder_lens_from_mask,
    mean_valid_abs_diff,
    mean_valid_frame_std,
)

from icefall import byte_encode, diagnostics, tokenize_by_CJK_char
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
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


def describe_cutset_size(cuts) -> str:
    try:
        return str(len(cuts))
    except TypeError:
        return "unknown (lazy CutSet)"


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


def _linear_ramp(batch_idx: int, start: int, steps: int, target: float) -> float:
    if target == 0.0:
        return 0.0
    if steps <= 0:
        return target if batch_idx >= start else 0.0
    if batch_idx <= start:
        return 0.0
    return target * min(1.0, (batch_idx - start) / steps)


class ConformerVIV2(nn.Module):
    """
    Conformer encoder + V-1 CTC head + posterior/prior blank gates.

    The original V-class encoder_output_layer is deliberately frozen and not
    used.  Checkpoints still use icefall's normal single `model` state_dict,
    with ctc_head/blank_gate/blank_prior as named submodules.
    """

    def __init__(
        self,
        conformer: Conformer,
        ctc_head: nn.Linear,
        blank_gate: BlankGateHeadV2,
        blank_prior: BlankPriorHeadV2,
    ) -> None:
        super().__init__()
        self.encoder = conformer
        self.ctc_head = ctc_head
        self.blank_gate = blank_gate
        self.blank_prior = blank_prior

    def forward(
        self,
        x: Tensor,
        supervisions: Optional[dict] = None,
        targets: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        warmup: float = 1.0,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor]:
        encoder_memory, memory_key_padding_mask = self.encoder.run_encoder(
            x, supervisions, warmup=warmup
        )
        encoder_hidden = encoder_memory.permute(1, 0, 2)
        encoder_out_lens = encoder_lens_from_mask(
            memory_key_padding_mask,
            batch_size=encoder_hidden.size(0),
            max_len=encoder_hidden.size(1),
            device=encoder_hidden.device,
        )

        log_p_nonblank = nn.functional.log_softmax(
            self.ctc_head(encoder_hidden), dim=-1
        )
        alpha_prior = self.blank_prior(encoder_hidden, encoder_out_lens)

        if targets is None or target_lengths is None:
            alpha_post = alpha_prior.new_zeros(alpha_prior.shape)
        else:
            alpha_post = self.blank_gate(
                encoder_hidden,
                targets,
                target_lengths,
                encoder_out_lens,
            )

        return (
            encoder_memory,
            memory_key_padding_mask,
            encoder_out_lens,
            log_p_nonblank,
            alpha_prior,
            alpha_post,
        )

    @property
    def device(self):
        return next(self.parameters()).device


# ── Per-frame helpers ──────────────────────────────────────────────────────────


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


def _valid_frame_mean(alpha: Tensor, valid_mask: Tensor) -> Tensor:
    mask = valid_mask.to(alpha.dtype)
    return (alpha * mask).sum() / mask.sum().clamp_min(1.0)


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
        "--exp-dir", type=str, default="conformer_ctc2/exp_vi_ot_v2",
        help="Experiment directory for checkpoints and logs."
    )
    parser.add_argument("--lang-dir", type=str, default="data/lang_phone")
    parser.add_argument(
        "--max-train-cuts",
        type=int,
        default=0,
        help="Use only the first N training cuts for smoke tests. 0 uses all cuts.",
    )
    parser.add_argument(
        "--max-valid-cuts",
        type=int,
        default=0,
        help="Use only the first N dev cuts for smoke tests. 0 uses all cuts.",
    )
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=2000,
        help="Run validation every this many training batches.",
    )
    parser.add_argument(
        "--validation-gate",
        type=str,
        default="train",
        choices=["train", "prior"],
        help=(
            "Gate used for validation checkpoint selection. 'train' preserves the "
            "training-time posterior/prior mixture; 'prior' matches inference."
        ),
    )
    parser.add_argument(
        "--validation-include-aux",
        type=str2bool,
        default=True,
        help=(
            "Include OT/KL/alpha auxiliary terms in validation loss. Set false "
            "with --validation-gate=prior to select checkpoints for prior-gate decode."
        ),
    )
    parser.add_argument("--initial-lr", type=float, default=0.003)
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for the acoustic encoder parameter group.",
    )
    parser.add_argument("--lr-batches", type=float, default=5000)
    parser.add_argument("--lr-epochs", type=float, default=6)
    parser.add_argument(
        "--init-encoder-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize the acoustic encoder from a VarCTC-v2 checkpoint. "
            "Token-dependent output heads and decoder parameters are excluded."
        ),
    )
    parser.add_argument(
        "--init-blank-prior",
        type=str2bool,
        default=False,
        help="Also initialize the token-independent blank prior from the checkpoint.",
    )
    parser.add_argument(
        "--att-rate", type=float, default=0.0,
        help="Attention decoder rate. Set 0.0 for pure CTC+VI."
    )
    parser.add_argument("--num-decoder-layers", type=int, default=0)
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--num-encoder-layers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-diagnostics", type=str2bool, default=False)
    parser.add_argument("--save-every-n", type=int, default=8000)
    parser.add_argument("--keep-last-k", type=int, default=20)
    parser.add_argument("--average-period", type=int, default=100)
    parser.add_argument("--use-fp16", type=str2bool, default=False)
    # --max-duration and other data args are provided by AishellAsrDataModule.

    # ── OT arguments ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--lambda-ot", type=float, default=0.1,
        help="Weight for the OT alignment loss (L_align)."
    )
    parser.add_argument(
        "--lambda-plan-w1",
        type=float,
        default=0.0,
        help=(
            "Weight for temporal W1 consistency that teaches a differentiable "
            "CTC token occupancy from a detached OT/FGW transport plan."
        ),
    )
    parser.add_argument(
        "--plan-w1-warmup-start",
        type=int,
        default=1000,
        help="Batch index at which to start ramping the OT/CTC plan consistency.",
    )
    parser.add_argument(
        "--plan-w1-warmup-steps",
        type=int,
        default=1000,
        help="Number of batches used to ramp --lambda-plan-w1 to its target.",
    )
    parser.add_argument(
        "--ot-eps", type=float, default=0.3,
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
    parser.add_argument(
        "--lambda-gw",
        type=float,
        default=0.0,
        help=(
            "FGW structural weight used to construct the detached transport "
            "plan. Zero exactly preserves the batched VFTA-OT path."
        ),
    )
    parser.add_argument(
        "--gw-n-outer",
        type=int,
        default=3,
        help="Number of FGW outer plan updates.",
    )
    parser.add_argument(
        "--lambda-gw-warmup-start",
        type=int,
        default=0,
        help="Global batch at which lambda_gw begins its linear ramp.",
    )
    parser.add_argument(
        "--lambda-gw-warmup-steps",
        type=int,
        default=0,
        help="Number of batches used to ramp lambda_gw to its target.",
    )

    # ── VI / blank gate arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--lambda-kl-blank", type=float, default=0.01,
        help="Weight for blank posterior-prior KL (L_KL^blank)."
    )
    parser.add_argument(
        "--lambda-alpha-mean",
        type=float,
        default=0.0,
        help=(
            "Optional hinge penalty on mean non-blank alpha. Default 0 disables it."
        ),
    )
    parser.add_argument(
        "--alpha-mean-target",
        type=float,
        default=0.35,
        help="Target value for the batch mean non-blank alpha hinge.",
    )
    parser.add_argument(
        "--alpha-mean-mode",
        type=str,
        default="ceiling",
        choices=["floor", "ceiling"],
        help=(
            "Use 'floor' to penalize alpha below target, or 'ceiling' to penalize "
            "alpha above target."
        ),
    )
    parser.add_argument(
        "--alpha-mean-source",
        type=str,
        default="eff",
        choices=["eff", "prior"],
        help=(
            "Which gate to regularize with --lambda-alpha-mean. 'eff' keeps the "
            "original behavior; 'prior' targets the X-only gate used by decode."
        ),
    )
    parser.add_argument(
        "--train-prior-mix",
        type=float,
        default=0.0,
        help=(
            "Keep this fraction of the X-only prior gate in the training-time "
            "CTC alpha after posterior warmup. This reduces posterior-train vs "
            "prior-decode mismatch. Default 0 keeps the original v2 behavior."
        ),
    )
    parser.add_argument(
        "--train-prior-logit-bias",
        type=float,
        default=0.0,
        help=(
            "Training-time calibration added to logit(alpha_prior) before "
            "building the gated CTC emissions. Use this to match the "
            "decode-time --prior-logit-bias."
        ),
    )
    parser.add_argument(
        "--train-prior-bias-start",
        type=int,
        default=0,
        help="Batch index at which the training prior logit-bias ramp starts.",
    )
    parser.add_argument(
        "--train-prior-bias-steps",
        type=int,
        default=0,
        help=(
            "Number of batches used to linearly ramp --train-prior-logit-bias. "
            "Set 0 to switch it on immediately at --train-prior-bias-start."
        ),
    )
    parser.add_argument(
        "--freeze-blank-prior",
        type=str2bool,
        default=False,
        help=(
            "Freeze the X-only blank prior head during fine-tuning. This is useful "
            "for train-time prior-bias calibration, where we want the rest of the "
            "model to adapt to the calibrated gate instead of letting the prior "
            "head immediately compensate the bias."
        ),
    )
    parser.add_argument(
        "--reset-optimizer-state",
        type=str2bool,
        default=False,
        help=(
            "When resuming from a checkpoint, initialize a fresh optimizer, Eden "
            "scheduler, and grad scaler instead of loading their checkpoint states."
        ),
    )
    parser.add_argument(
        "--bias-only-calibration",
        type=str2bool,
        default=False,
        help=(
            "Freeze the whole model except blank_prior.linear.bias. This is a "
            "controlled calibration mode for absorbing a decode-time prior bias "
            "without letting the encoder/CTC head adapt."
        ),
    )
    parser.add_argument(
        "--blank-prior-bias-init-shift",
        type=float,
        default=0.0,
        help=(
            "After loading the checkpoint, add this scalar to "
            "blank_prior.linear.bias before bias-only calibration. Setting this "
            "to the dev-selected decode bias bakes that calibration into the "
            "checkpoint and then optionally fine-tunes the scalar."
        ),
    )
    parser.add_argument(
        "--alpha-smooth-mix", type=float, default=0.1,
        help=(
            "Uniform mix weight for OT row marginal smoothing. "
            "Prevents a sharply peaked blank gate from making OT trivial."
        ),
    )
    parser.add_argument(
        "--col-marginal-type", type=str, default="acoustic",
        choices=["bpe", "acoustic", "uniform"],
        help=(
            "'bpe': column marginal proportional to BPE token character length "
            "(theta-independent linguistic prior). "
            "'acoustic': previous detached diagonal/acoustic token prior. "
            "'uniform': b_u = 1/U."
        ),
    )
    parser.add_argument(
        "--bpe-col-floor", type=float, default=0.05,
        help="Uniform floor for BPE-length column marginal (keeps all tokens reachable)."
    )
    parser.add_argument(
        "--ot-token-prior-sigma",
        type=float,
        default=0.15,
        help="Width of the soft diagonal mask for acoustic column marginals.",
    )
    parser.add_argument(
        "--ot-token-prior-score-temp",
        type=float,
        default=1.0,
        help="Temperature for acoustic scores used in acoustic column marginals.",
    )
    parser.add_argument(
        "--ot-token-prior-floor",
        type=float,
        default=0.05,
        help="Uniform floor mixed into acoustic column marginals.",
    )

    # ── Blank gate architecture ────────────────────────────────────────────────
    parser.add_argument(
        "--label-embed-dim", type=int, default=256,
        help="Dimension for posterior cross-attention label embeddings.",
    )
    parser.add_argument(
        "--init-blank-prob", type=float, default=0.1,
        help=(
            "Initial non-blank probability for both prior and posterior gates "
            "(sigmoid(bias) at init). Lower values bias toward blank-heavy CTC."
        ),
    )

    # ── Posterior warm-up arguments ───────────────────────────────────────────
    parser.add_argument(
        "--gate-warmup-start", type=int, default=2000,
        help=(
            "Batch index at which to start mixing the posterior cross-attention gate. "
            "Before this, alpha_eff = alpha_prior only (stable prior-only phase)."
        ),
    )
    parser.add_argument(
        "--gate-warmup-steps", type=int, default=4000,
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
            "valid_interval": 2000,
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


def initialize_from_pretrained_encoder(
    model: nn.Module,
    filename: Path,
    include_blank_prior: bool = False,
) -> None:
    """Load a tokenization-independent acoustic backbone from a VI checkpoint."""
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()

    def should_transfer(name: str) -> bool:
        acoustic_encoder = (
            name.startswith("encoder.")
            and not name.startswith("encoder.encoder_output_layer.")
            and not name.startswith("encoder.decoder.")
        )
        return acoustic_encoder or (
            include_blank_prior and name.startswith("blank_prior.")
        )

    required = [name for name in target if should_transfer(name)]
    incompatible = [
        name
        for name in required
        if name not in source or source[name].shape != target[name].shape
    ]
    if incompatible:
        preview = ", ".join(incompatible[:8])
        raise ValueError(
            "Pretrained encoder is incompatible with the target architecture. "
            f"First incompatible keys: {preview}"
        )

    transferred = {name: source[name] for name in required}
    model.load_state_dict(transferred, strict=False)
    num_params = sum(value.numel() for value in transferred.values())
    logging.info(
        "Initialized %s tensors (%.2fM parameters) from %s%s",
        len(transferred),
        num_params / 1.0e6,
        filename,
        " including blank_prior" if include_blank_prior else "",
    )


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


def _effective_lambda_gw(params: AttributeDict) -> float:
    return _linear_ramp(
        batch_idx=int(params.batch_idx_train),
        start=int(params.lambda_gw_warmup_start),
        steps=int(params.lambda_gw_warmup_steps),
        target=float(params.lambda_gw),
    )


def _encoder_aligned_acoustic_features(
    feature: Tensor,
    feature_lens: Tensor,
    original_index: int,
    encoder_length: int,
    subsampling_factor: int,
) -> Tensor:
    """Match the raw-Fbank sampling convention used by LibriSpeech FGW."""
    raw_length = int(feature_lens[original_index].item())
    indices = (
        torch.arange(encoder_length, device=feature.device) * subsampling_factor
        + subsampling_factor // 2
    )
    indices = indices.clamp(max=max(raw_length - 1, 0))
    return feature[original_index].index_select(0, indices)


# ── Core training loss ─────────────────────────────────────────────────────────


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    graph_compiler: Union[
        BpeCtcTrainingGraphCompiler,
        CharCtcTrainingGraphCompiler,
        CtcTrainingGraphCompiler,
    ],
    is_training: bool,
    warmup: float = 1.0,
    debug: bool = False,
) -> Tuple[Tensor, MetricsTracker, Optional[Dict[str, Any]]]:
    """
    VarCTC v2 loss:
      L = (1-att_rate) L_CTC_gated + att_rate L_AED
          + lambda_kl L_KL^blank + lambda_ot L_align
          + lambda_plan_w1 L_W1(P_OT, A_CTC)
    """
    if not isinstance(
        graph_compiler,
        (
            BpeCtcTrainingGraphCompiler,
            CharCtcTrainingGraphCompiler,
            PhoneCtcTrainingGraphCompiler,
        ),
    ):
        raise ValueError("VarCTC v2 requires BPE, char, or phone targets.")

    model_ref = model.module if hasattr(model, "module") else model
    device = next(model_ref.parameters()).device
    feature = batch["inputs"].to(device)
    assert feature.ndim == 3

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )
    token_ids = graph_compiler.texts_to_ids(texts)
    decoding_graph = graph_compiler.compile(token_ids)

    targets_padded, target_lengths_t = _build_padded_targets(token_ids, device)
    sorted_to_orig = supervision_segments[:, 0].tolist()
    orig_order = torch.zeros(len(token_ids), dtype=torch.long, device=device)
    for sorted_idx, orig_idx in enumerate(sorted_to_orig):
        orig_order[orig_idx] = sorted_idx
    targets_orig = targets_padded[orig_order]
    tlen_orig = target_lengths_t[orig_order]

    # Posterior warm-up mix: ramp from 0 (prior-only) → 1 (full posterior)
    if not is_training and params.validation_gate == "prior":
        mix = 0.0
    else:
        mix = _gate_mix(
            batch_idx=params.batch_idx_train,
            warmup_start=params.gate_warmup_start,
            warmup_steps=params.gate_warmup_steps,
        )

    with torch.set_grad_enabled(is_training):
        (
            encoder_memory,
            memory_mask,
            encoder_out_lens,
            log_p_nonblank,
            alpha_prior_batch,
            alpha_post_batch,
        ) = model(
            feature,
            supervisions,
            targets=targets_orig,
            target_lengths=tlen_orig,
            warmup=warmup,
        )

        train_prior_logit_bias = _linear_ramp(
            batch_idx=params.batch_idx_train,
            start=params.train_prior_bias_start,
            steps=params.train_prior_bias_steps,
            target=float(params.train_prior_logit_bias),
        )
        if train_prior_logit_bias != 0.0:
            alpha_prior_batch = alpha_prior_batch.float().clamp(1.0e-5, 1.0 - 1.0e-5)
            alpha_prior_batch = torch.sigmoid(
                torch.logit(alpha_prior_batch) + train_prior_logit_bias
            ).to(log_p_nonblank.dtype)

        # At mix=0 this is prior-only; after warmup, keep an optional fraction
        # of prior in the CTC gate so the inference-time gate receives direct
        # CTC training instead of relying only on KL to chase the posterior.
        train_prior_mix = min(max(float(params.train_prior_mix), 0.0), 1.0)
        post_weight = mix * (1.0 - train_prior_mix)
        prior_weight = 1.0 - post_weight
        alpha_eff_batch = prior_weight * alpha_prior_batch + post_weight * alpha_post_batch
        nnet_output_gated = build_gated_log_probs_v2(
            log_p_nonblank,
            alpha_eff_batch,
        )

    T_out = log_p_nonblank.size(1)
    frame_idx = torch.arange(T_out, device=device).unsqueeze(0)
    valid_mask = frame_idx < encoder_out_lens.unsqueeze(1)

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

    # ── KL and OT losses (batched over the whole minibatch — see ot_prior_v2) ──
    # The old per-utterance Python loop was launch/sync bound: each short AISHELL
    # utterance triggered a .item() sync, host→device copies and a full Sinkhorn
    # of tiny matrices.  We now reorder the batch into sorted (token_ids) order
    # and compute KL and OT for the whole minibatch in one vectorized call.
    include_aux = is_training or params.validation_include_aux
    lambda_ot = params.lambda_ot if include_aux else 0.0
    lambda_kl = params.lambda_kl_blank if include_aux else 0.0
    lambda_alpha_mean = params.lambda_alpha_mean if include_aux else 0.0
    lambda_plan_w1 = (
        _linear_ramp(
            batch_idx=params.batch_idx_train,
            start=params.plan_w1_warmup_start,
            steps=params.plan_w1_warmup_steps,
            target=float(params.lambda_plan_w1),
        )
        if is_training
        else 0.0
    )
    debug_info: Optional[Dict[str, Any]] = None

    # targets_padded / target_lengths_t are already in sorted (token_ids) order;
    # reorder the model outputs to match so the loss vectors line up with ctc_loss.
    sorted_index = torch.tensor(sorted_to_orig, device=device, dtype=torch.long)
    lp_sorted = log_p_nonblank[sorted_index]          # [B, T, V-1]
    alpha_eff_sorted = alpha_eff_batch[sorted_index]  # [B, T]
    alpha_post_sorted = alpha_post_batch[sorted_index]
    alpha_prior_sorted = alpha_prior_batch[sorted_index]
    flen_sorted = encoder_out_lens[sorted_index]      # [B]
    labels_sorted = targets_padded                    # [B, U]
    llen_sorted = target_lengths_t                    # [B]

    T_max = lp_sorted.size(1)
    valid_mask_sorted = (
        torch.arange(T_max, device=device)[None, :] < flen_sorted[:, None]
    )

    # KL(Bern(post.detach()) || Bern(prior)), per-frame mean over valid frames.
    # (Vectorized form of _blank_kl_loss; posterior is stop-grad, prior chases it.)
    if is_training and mix > 0.0:
        kl_eps = 1.0e-5
        p = alpha_post_sorted.detach().float().clamp(kl_eps, 1.0 - kl_eps)
        q = alpha_prior_sorted.float().clamp(kl_eps, 1.0 - kl_eps)
        kl_per_frame = p * (p.log() - q.log()) + (1.0 - p) * (
            (1.0 - p).log() - (1.0 - q).log()
        )
        m = valid_mask_sorted.float()
        kl_loss = (kl_per_frame * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    else:
        kl_loss = lp_sorted.new_zeros(lp_sorted.size(0))

    # OT alignment loss, vectorized across the minibatch.
    if lambda_ot > 0 or lambda_plan_w1 > 0:
        if params.col_marginal_type == "bpe":
            if not isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
                raise ValueError(
                    "--col-marginal-type=bpe requires a BPE/BBPE lang dir."
                )
            U_max = labels_sorted.size(1)
            bpe_rows = []
            for ids in token_ids:
                row = _compute_bpe_lengths(ids, graph_compiler.sp, device)
                if row.numel() < U_max:
                    row = torch.cat([row, row.new_ones(U_max - row.numel())], dim=0)
                bpe_rows.append(row)
            bpe_lengths_b = torch.stack(bpe_rows, dim=0)
        else:
            bpe_lengths_b = None

        effective_lambda_gw = _effective_lambda_gw(params)
        if effective_lambda_gw > 0.0:
            # Pad the encoder-aligned raw Fbanks, then solve every utterance's
            # independent FGW problem in one batched set of GPU kernels.
            raw_sorted = feature[sorted_index]
            raw_lens_sorted = feature_lens[sorted_index]
            acoustic_indices = (
                torch.arange(T_max, device=device) * params.subsampling_factor
                + params.subsampling_factor // 2
            )[None, :]
            acoustic_indices = torch.minimum(
                acoustic_indices,
                (raw_lens_sorted - 1).clamp_min(0)[:, None],
            )
            acoustic_features_b = torch.gather(
                raw_sorted,
                dim=1,
                index=acoustic_indices[:, :, None].expand(
                    -1, -1, raw_sorted.size(2)
                )
            )
            need_plan = lambda_plan_w1 > 0 or debug
            fgw_result = vi_fgw_loss_v2_batched(
                log_p_nonblank=lp_sorted,
                alpha=alpha_eff_sorted,
                labels=labels_sorted,
                frame_lens=flen_sorted,
                label_lens=llen_sorted,
                acoustic_features=acoustic_features_b,
                bpe_lengths=bpe_lengths_b,
                column_marginal_type=params.col_marginal_type,
                alpha_smooth_mix=params.alpha_smooth_mix,
                bpe_col_floor=params.bpe_col_floor,
                token_prior_sigma=params.ot_token_prior_sigma,
                token_prior_score_temp=params.ot_token_prior_score_temp,
                token_prior_floor=params.ot_token_prior_floor,
                eps=params.ot_eps,
                iters=params.ot_iters,
                beta_pos=params.ot_beta_pos,
                lambda_gw=effective_lambda_gw,
                n_outer=params.gw_n_outer,
                return_plan=need_plan,
            )
            if need_plan:
                ot_loss, consistency_plan = fgw_result
            else:
                ot_loss = fgw_result
                consistency_plan = None
            if debug and consistency_plan is not None:
                if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
                    token_pieces = [
                        graph_compiler.sp.id_to_piece(token) for token in token_ids[0]
                    ]
                else:
                    token_pieces = [
                        graph_compiler.token_table[token] for token in token_ids[0]
                    ]
                cuts = supervisions.get("cut")
                original_index = int(sorted_to_orig[0])
                cut_id = (
                    getattr(cuts[original_index], "id", None)
                    if cuts is not None
                    else None
                )
                debug_info = {
                    "P": consistency_plan[0, : int(flen_sorted[0]), : int(llen_sorted[0])],
                    "token_pieces": token_pieces,
                    "cut_id": cut_id,
                }
        else:
            ot_result = vi_ot_loss_v2_batched(
                log_p_nonblank=lp_sorted,
                alpha=alpha_eff_sorted,
                labels=labels_sorted,
                frame_lens=flen_sorted,
                label_lens=llen_sorted,
                bpe_lengths=bpe_lengths_b,
                column_marginal_type=params.col_marginal_type,
                alpha_smooth_mix=params.alpha_smooth_mix,
                bpe_col_floor=params.bpe_col_floor,
                token_prior_sigma=params.ot_token_prior_sigma,
                token_prior_score_temp=params.ot_token_prior_score_temp,
                token_prior_floor=params.ot_token_prior_floor,
                eps=params.ot_eps,
                iters=params.ot_iters,
                beta_pos=params.ot_beta_pos,
                return_plan=lambda_plan_w1 > 0,
                differentiable_plan=False,
            )
            if lambda_plan_w1 > 0:
                ot_loss, consistency_plan = ot_result
            else:
                ot_loss = ot_result
                consistency_plan = None

        if lambda_plan_w1 > 0:
            assert consistency_plan is not None
            ctc_occupancy = ctc_token_occupancy_batched(
                log_probs=nnet_output_gated[sorted_index],
                labels=labels_sorted,
                frame_lens=flen_sorted,
                label_lens=llen_sorted,
                blank_id=0,
            )
            plan_w1_loss = temporal_plan_w1_loss(
                student_occupancy=ctc_occupancy,
                teacher_plan=consistency_plan,
                frame_lens=flen_sorted,
                label_lens=llen_sorted,
            )
        else:
            plan_w1_loss = lp_sorted.new_zeros(lp_sorted.size(0))

        if debug and effective_lambda_gw <= 0.0:
            # Recompute the transport plan for the first utterance only (viz).
            ids0 = token_ids[0]
            L0 = int(flen_sorted[0].item())
            _, P = vi_ot_loss_v2(
                log_p_nonblank=lp_sorted[0, :L0],
                alpha=alpha_eff_sorted[0, :L0],
                labels=torch.tensor(ids0, device=device, dtype=torch.long),
                bpe_lengths=(
                    bpe_lengths_b[0, : len(ids0)] if bpe_lengths_b is not None else None
                ),
                column_marginal_type=params.col_marginal_type,
                alpha_smooth_mix=params.alpha_smooth_mix,
                bpe_col_floor=params.bpe_col_floor,
                token_prior_sigma=params.ot_token_prior_sigma,
                token_prior_score_temp=params.ot_token_prior_score_temp,
                token_prior_floor=params.ot_token_prior_floor,
                eps=params.ot_eps,
                iters=params.ot_iters,
                beta_pos=params.ot_beta_pos,
                return_plan=True,
            )
            if P is not None:
                if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
                    token_pieces = [graph_compiler.sp.id_to_piece(t) for t in ids0]
                else:
                    token_pieces = [graph_compiler.token_table[t] for t in ids0]
                cuts = supervisions.get("cut", None)
                cut_id = None
                orig0 = sorted_to_orig[0]
                if cuts is not None and orig0 < len(cuts):
                    cut_id = getattr(cuts[orig0], "id", None)
                debug_info = {"P": P, "token_pieces": token_pieces, "cut_id": cut_id}
    else:
        ot_loss = lp_sorted.new_zeros(lp_sorted.size(0))
        plan_w1_loss = lp_sorted.new_zeros(lp_sorted.size(0))
    alpha_mean_source = getattr(params, "alpha_mean_source", "eff")
    if alpha_mean_source == "prior":
        alpha_mean_for_loss = alpha_prior_batch.float()
    else:
        alpha_mean_for_loss = alpha_eff_batch.float()
    alpha_mean = _valid_frame_mean(alpha_mean_for_loss, valid_mask)
    alpha_mean_target = alpha_mean.new_tensor(float(params.alpha_mean_target))
    alpha_mean_mode = getattr(params, "alpha_mean_mode", "ceiling")
    if alpha_mean_mode == "floor":
        alpha_mean_loss = torch.relu(alpha_mean_target - alpha_mean)
    elif alpha_mean_mode == "ceiling":
        alpha_mean_loss = torch.relu(alpha_mean - alpha_mean_target)
    else:
        raise ValueError(f"Unsupported alpha_mean_mode: {alpha_mean_mode}")
    alpha_mean_loss_total = alpha_mean_loss * valid_mask.sum().to(alpha_mean_loss.dtype)

    if not torch.all(ctc_loss_is_finite):
        logging.info("Not all CTC losses are finite!")
        ctc_loss = ctc_loss[ctc_loss_is_finite]
        kl_loss = kl_loss[ctc_loss_is_finite]
        ot_loss = ot_loss[ctc_loss_is_finite]
        plan_w1_loss = plan_w1_loss[ctc_loss_is_finite]
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
            + lambda_plan_w1 * plan_w1_loss.sum()
            + lambda_alpha_mean * alpha_mean_loss_total
        )
    else:
        loss = (
            (1.0 - params.att_rate) * ctc_loss.sum()
            + lambda_kl * kl_loss.sum()
            + lambda_ot * ot_loss.sum()
            + lambda_plan_w1 * plan_w1_loss.sum()
            + lambda_alpha_mean * alpha_mean_loss_total
        )
        att_loss = log_p_nonblank.new_tensor(0.0)

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = encoder_out_lens.sum().detach().cpu().item()
    valid_frames = valid_mask.sum().clamp_min(1).float()
    metric_frames = info["frames"]
    info["ctc_loss"] = ctc_loss.sum().detach().cpu().item()
    info["kl_blank_loss"] = kl_loss.sum().detach().cpu().item()
    info["ot_loss"] = ot_loss.sum().detach().cpu().item()
    info["lambda_gw"] = _effective_lambda_gw(params) * metric_frames
    info["lambda_gw_target"] = float(params.lambda_gw) * metric_frames
    info["gw_n_outer"] = float(params.gw_n_outer) * metric_frames
    info["plan_w1_loss"] = plan_w1_loss.sum().detach().cpu().item()
    info["lambda_plan_w1"] = lambda_plan_w1 * metric_frames
    info["alpha_mean_loss"] = alpha_mean_loss.detach().cpu().item() * metric_frames
    info["gate_mix"] = mix * metric_frames
    info["train_prior_mix"] = (
        min(max(float(params.train_prior_mix), 0.0), 1.0) * metric_frames
    )
    info["train_prior_logit_bias"] = train_prior_logit_bias * metric_frames
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
    info["alpha_prior_frame_std"] = (
        mean_valid_frame_std(alpha_prior_batch, encoder_out_lens)
        .detach()
        .cpu()
        .item()
        * metric_frames
    )
    info["alpha_post_frame_std"] = (
        mean_valid_frame_std(alpha_post_batch, encoder_out_lens)
        .detach()
        .cpu()
        .item()
        * metric_frames
    )
    info["alpha_eff_frame_std"] = (
        mean_valid_frame_std(alpha_eff_batch, encoder_out_lens)
        .detach()
        .cpu()
        .item()
        * metric_frames
    )
    info["alpha_prior_post_absdiff"] = (
        mean_valid_abs_diff(alpha_post_batch, alpha_prior_batch, encoder_out_lens)
        .detach()
        .cpu()
        .item()
        * metric_frames
    )
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
    graph_compiler: Union[
        BpeCtcTrainingGraphCompiler,
        CharCtcTrainingGraphCompiler,
        CtcTrainingGraphCompiler,
    ],
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

    if tot_loss["frames"] == 0:
        raise RuntimeError(
            "No training batches were produced. This usually happens in a "
            "small smoke run when DynamicBucketingSampler uses drop_last=True. "
            "Try --drop-last false, increase --max-train-cuts, or increase "
            "--max-duration."
        )

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
    graph_compiler: Union[
        BpeCtcTrainingGraphCompiler,
        CharCtcTrainingGraphCompiler,
        CtcTrainingGraphCompiler,
    ],
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

        if (
            params.valid_interval > 0
            and params.batch_idx_train % params.valid_interval == 0
        ):
            logging.info("Computing validation loss")
            previous_best_valid_loss = params.best_valid_loss
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            if rank == 0 and params.best_valid_loss < previous_best_valid_loss:
                best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                logging.info(
                    "Saving best validation checkpoint at batch %s to %s",
                    params.batch_idx_train,
                    best_valid_filename,
                )
                save_checkpoint_impl(
                    filename=best_valid_filename,
                    model=model,
                    model_avg=model_avg,
                    params=params,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=train_dl.sampler,
                    scaler=scaler,
                    rank=rank,
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

    if not 0.0 <= params.lambda_gw <= 1.0:
        raise ValueError("--lambda-gw must be between 0 and 1")
    if params.gw_n_outer < 0:
        raise ValueError("--gw-n-outer must be non-negative")

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

    if "lang_bpe" in str(params.lang_dir) or "lang_bbpe" in str(params.lang_dir):
        graph_compiler = BpeCtcTrainingGraphCompiler(
            params.lang_dir,
            device=device,
            sos_token="<sos/eos>",
            eos_token="<sos/eos>",
        )
        text_is_byte_bpe = True
    elif "lang_char" in str(params.lang_dir):
        graph_compiler = CharCtcTrainingGraphCompiler(
            lexicon=lexicon,
            device=device,
            sos_token="<sos/eos>",
            eos_token="<sos/eos>",
        )
        text_is_byte_bpe = False
    elif "lang_phone" in str(params.lang_dir):
        # TIMIT phones: supervision.text is a space-separated phone sequence.
        graph_compiler = PhoneCtcTrainingGraphCompiler(
            lexicon=lexicon,
            device=device,
        )
        text_is_byte_bpe = False
    else:
        raise ValueError(
            "VarCTC v2 requires a lang_bpe/lang_bbpe/lang_char/lang_phone directory."
        )

    logging.info("Building ConformerVIV2 model")
    conformer = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )
    for p in conformer.encoder_output_layer.parameters():
        p.requires_grad_(False)
    if params.att_rate == 0.0 and hasattr(conformer, "decoder"):
        for p in conformer.decoder.parameters():
            p.requires_grad_(False)

    ctc_head = nn.Linear(params.encoder_dim, num_classes - 1)
    nn.init.xavier_uniform_(ctc_head.weight)
    nn.init.zeros_(ctc_head.bias)

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
    model = ConformerVIV2(conformer, ctc_head, blank_gate, blank_prior)

    if params.init_encoder_checkpoint is not None:
        if params.start_epoch > 1 or params.start_batch > 0:
            raise ValueError(
                "--init-encoder-checkpoint initializes a new experiment and cannot "
                "be combined with --start-epoch > 1 or --start-batch > 0."
            )
        initialize_from_pretrained_encoder(
            model,
            filename=params.init_encoder_checkpoint,
            include_blank_prior=params.init_blank_prior,
        )

    if params.freeze_blank_prior:
        logging.info("Freezing blank_prior parameters")
        for p in model.blank_prior.parameters():
            p.requires_grad_(False)

    if params.bias_only_calibration:
        logging.info("Bias-only calibration: freezing all parameters")
        for p in model.parameters():
            p.requires_grad_(False)
        model.blank_prior.linear.bias.requires_grad_(True)

    print(model)

    num_param = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of model parameters: {num_param}")
    logging.info(f"Number of trainable model parameters: {num_trainable}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        model_avg = copy.deepcopy(model)

    assert params.start_epoch > 0
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    if params.blank_prior_bias_init_shift != 0.0:
        logging.info(
            "Adding %.6g to blank_prior.linear.bias after checkpoint load",
            params.blank_prior_bias_init_shift,
        )
        with torch.no_grad():
            model.blank_prior.linear.bias.add_(params.blank_prior_bias_init_shift)
            if model_avg is not None:
                model_avg.blank_prior.linear.bias.add_(
                    params.blank_prior_bias_init_shift
                )

    if params.bias_only_calibration:
        bias_value = model.blank_prior.linear.bias.detach().mean().item()
        logging.info(
            "Bias-only calibration trainable parameter: "
            "blank_prior.linear.bias, initial mean %.6g",
            bias_value,
        )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank])

    if params.encoder_lr_scale <= 0:
        raise ValueError("--encoder-lr-scale must be positive")
    if params.encoder_lr_scale == 1.0:
        optimizer_params = [p for p in model.parameters() if p.requires_grad]
    else:
        encoder_params = []
        head_params = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(parameter)
            else:
                head_params.append(parameter)
        optimizer_params = [
            {
                "params": encoder_params,
                "lr": params.initial_lr * params.encoder_lr_scale,
            },
            {"params": head_params, "lr": params.initial_lr},
        ]
        logging.info(
            "Using differential learning rates: encoder=%.6g, heads=%.6g",
            params.initial_lr * params.encoder_lr_scale,
            params.initial_lr,
        )
    optimizer = Eve(optimizer_params, lr=params.initial_lr)
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if params.reset_optimizer_state:
        logging.info("Reset optimizer/scheduler state requested; using fresh states")
    elif checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        not params.reset_optimizer_state
        and checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        diagnostic = diagnostics.attach_diagnostics(model)

    timit = TimitAsrDataModule(args)
    train_cuts = timit.train_cuts()
    valid_cuts = timit.valid_cuts()

    def remove_short_and_long_utt(c: Cut):
        return 0.5 <= c.duration <= 15.0

    def tokenize_and_encode_text(c: Cut):
        text = c.supervisions[0].text
        c.supervisions[0].text = byte_encode(tokenize_by_CJK_char(text))
        return c

    def remove_invalid_utt_ctc(c: Cut):
        # texts_to_ids expects a list of transcripts and returns list-of-lists.
        token_ids = graph_compiler.texts_to_ids([c.supervisions[0].text])[0]
        num_tokens = len(token_ids)
        if num_tokens == 0:
            return False
        adjacent_repeats = sum(a == b for a, b in zip(token_ids, token_ids[1:]))
        min_ctc_frames = num_tokens + adjacent_repeats
        num_frames = c.features.num_frames
        num_subsampled = ((num_frames - 1) // 2 - 1) // 2
        return min_ctc_frames <= num_subsampled

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    valid_cuts = valid_cuts.filter(remove_short_and_long_utt)
    if text_is_byte_bpe:
        train_cuts = train_cuts.map(tokenize_and_encode_text)
        valid_cuts = valid_cuts.map(tokenize_and_encode_text)
    train_cuts = train_cuts.filter(remove_invalid_utt_ctc)
    valid_cuts = valid_cuts.filter(remove_invalid_utt_ctc)

    if params.max_train_cuts > 0:
        logging.info("Using only the first %s training cuts", params.max_train_cuts)
        train_cuts = train_cuts.subset(first=params.max_train_cuts)

    if params.max_valid_cuts > 0:
        logging.info("Using only the first %s validation cuts", params.max_valid_cuts)
        valid_cuts = valid_cuts.subset(first=params.max_valid_cuts)

    logging.info(
        "Number of training cuts after filtering: %s",
        describe_cutset_size(train_cuts),
    )
    logging.info(
        "Number of validation cuts after filtering: %s",
        describe_cutset_size(valid_cuts),
    )

    if params.max_train_cuts > 0 and args.drop_last:
        logging.info(
            "Disabling drop_last for a small smoke run with --max-train-cuts=%s",
            params.max_train_cuts,
        )
        args.drop_last = False
        params.drop_last = False
    if (
        params.max_train_cuts > 0
        and args.bucketing_sampler
        and args.num_buckets > params.max_train_cuts
    ):
        logging.info(
            "Reducing num_buckets from %s to %s for the small smoke run",
            args.num_buckets,
            params.max_train_cuts,
        )
        args.num_buckets = params.max_train_cuts
        params.num_buckets = params.max_train_cuts

    train_dl = timit.train_dataloaders(train_cuts)
    valid_dl = timit.valid_dataloaders(valid_cuts)

    scaler = create_grad_scaler(enabled=params.use_fp16)
    if params.reset_optimizer_state:
        logging.info("Reset grad scaler state requested; using a fresh scaler")
    elif checkpoints and "grad_scaler" in checkpoints:
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
    TimitAsrDataModule.add_arguments(parser)
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
