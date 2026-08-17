"""Conformer wrapper and helpers for label-prior CTC."""

import math
from typing import Dict

import torch
from torch import Tensor

from conformer import Conformer as BaseConformer


def apply_label_prior(
    log_probs: Tensor,
    label_prior: Tensor,
    alpha: float,
    floor: float,
    enabled: bool = True,
) -> Tensor:
    """Apply the label-prior CTC path score y_k^t / P(k)^alpha in log space."""
    if alpha == 0.0 or not enabled:
        return log_probs

    prior = label_prior.to(device=log_probs.device, dtype=torch.float32)
    prior = prior.clamp_min(floor)
    prior = prior / prior.sum()
    prior_penalty = float(alpha) * prior.log()
    return log_probs - prior_penalty.to(log_probs.dtype).view(1, 1, -1)


class LabelPriorConformer(BaseConformer):
    """Baseline Conformer plus epoch-level unigram label-prior statistics."""

    # Decode-time controls. Off by default so training keeps the raw CTC
    # output (the training prior is applied separately in compute_loss). The
    # decoder sets these per instance to apply log y - alpha * log P(k) at
    # inference, matching Huang et al. (priors are applied during Viterbi too).
    apply_prior_in_forward: bool = False
    decode_alpha: float = 0.0
    decode_floor: float = math.exp(-12.0)

    def __init__(self, *args, **kwargs) -> None:
        num_classes = int(kwargs["num_classes"])
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "label_prior",
            torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32),
        )
        self.register_buffer(
            "label_prior_sum",
            torch.zeros(num_classes, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "label_prior_total",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("label_prior_ready", torch.tensor(False))

    def forward(self, *args, **kwargs):
        nnet_output, memory, memory_key_padding_mask = super().forward(
            *args, **kwargs
        )
        if self.apply_prior_in_forward and float(self.decode_alpha) > 0.0:
            nnet_output = apply_label_prior(
                nnet_output,
                self.label_prior,
                float(self.decode_alpha),
                float(self.decode_floor),
                enabled=bool(self.label_prior_ready.item()),
            )
        return nnet_output, memory, memory_key_padding_mask

    def prior_adjusted_log_probs(
        self,
        log_probs: Tensor,
        alpha: float,
        floor: float,
        enabled: bool = True,
    ):
        return apply_label_prior(
            log_probs,
            self.label_prior,
            alpha,
            floor,
            enabled=enabled and bool(self.label_prior_ready.item()),
        )

    @torch.no_grad()
    def reset_label_prior_stats(self) -> None:
        self.label_prior_sum.zero_()
        self.label_prior_total.zero_()

    @torch.no_grad()
    def accumulate_label_prior_stats(
        self,
        log_probs: Tensor,
        supervision_segments: Tensor,
    ) -> None:
        """Accumulate posteriorgram sums over non-padded supervised frames."""
        probs = log_probs.detach().float().exp()
        total_frames = 0
        for sequence_idx, start_frame, num_frames in supervision_segments.tolist():
            seq = int(sequence_idx)
            start = int(start_frame)
            end = start + int(num_frames)
            if end <= start:
                continue
            segment = probs[seq, start:end]
            self.label_prior_sum.add_(segment.sum(dim=0))
            total_frames += segment.size(0)

        if total_frames > 0:
            self.label_prior_total.add_(
                torch.tensor(
                    float(total_frames),
                    device=self.label_prior_total.device,
                    dtype=self.label_prior_total.dtype,
                )
            )

    @torch.no_grad()
    def sync_and_update_label_prior(
        self,
        momentum: float,
        floor: float,
    ) -> Dict[str, float]:
        """All-reduce epoch stats and replace/EMA-update the saved prior."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                self.label_prior_sum, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                self.label_prior_total, op=torch.distributed.ReduceOp.SUM
            )

        total = float(self.label_prior_total.item())
        if total > 0.0:
            new_prior = self.label_prior_sum / self.label_prior_total.clamp_min(1.0)
            new_prior = new_prior.clamp_min(floor)
            new_prior = new_prior / new_prior.sum()
            if momentum > 0.0:
                old_prior = self.label_prior.float()
                new_prior = momentum * old_prior + (1.0 - momentum) * new_prior
                new_prior = new_prior.clamp_min(floor)
                new_prior = new_prior / new_prior.sum()
            self.label_prior.copy_(new_prior.to(self.label_prior.dtype))
            self.label_prior_ready.fill_(True)

        prior = self.label_prior.float().clamp_min(floor)
        prior = prior / prior.sum()
        entropy = -(prior * prior.log()).sum().item()
        stats = {
            "frames": total,
            "ready": float(self.label_prior_ready.item()),
            "blank": prior[0].item(),
            "min": prior.min().item(),
            "max": prior.max().item(),
            "entropy": entropy,
        }
        self.reset_label_prior_stats()
        return stats
