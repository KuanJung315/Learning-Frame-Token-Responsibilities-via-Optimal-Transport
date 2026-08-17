"""Stride-2 5M TDNN-FFN CTC model used by the Label-Prior comparison.

The topology follows the public Label-Prior recipe: three TDNN layers with
specifications (5, 2, 1), (3, 1, 1), (3, 1, 1), followed by a five-layer
residual FFN.  The wrapper exposes the interface expected by the local
``conformer_ctc2`` trainer while retaining an epoch-level label-prior buffer.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def apply_label_prior(
    log_probs: Tensor,
    label_prior: Tensor,
    alpha: float,
    floor: float,
    enabled: bool = True,
) -> Tensor:
    """Return unnormalised CTC path scores log y - alpha log P(label)."""
    if float(alpha) == 0.0 or not enabled:
        return log_probs
    prior = label_prior.to(device=log_probs.device, dtype=torch.float32)
    prior = prior.clamp_min(float(floor))
    prior = prior / prior.sum()
    return log_probs - (
        float(alpha) * prior.log()
    ).to(log_probs.dtype).view(1, 1, -1)


def _conv_output_lengths(
    lengths: Tensor, kernel_size: int, stride: int, dilation: int, padding: int
) -> Tensor:
    lengths = lengths.to(torch.long)
    numerator = lengths + 2 * padding - dilation * (kernel_size - 1) - 1
    return torch.div(numerator, stride, rounding_mode="floor") + 1


class TdnnBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.padding = ((self.kernel_size - 1) // 2) * self.dilation
        layers = [
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding,
            ),
            nn.ReLU(inplace=True),
            # affine=False is part of the public 5M reference topology.
            nn.BatchNorm1d(output_dim, affine=False),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        valid = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        x = x * valid.unsqueeze(-1).to(x.dtype)
        x = self.layers(x.transpose(1, 2)).transpose(1, 2)
        lengths = _conv_output_lengths(
            lengths,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
        )
        return x, lengths


class ResidualFfn(nn.Module):
    def __init__(self, dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


class TdnnLabelPriorCTC(nn.Module):
    """5M TDNN model compatible with the icefall CTC training loop."""

    apply_prior_in_forward: bool = False
    decode_alpha: float = 0.0
    decode_floor: float = math.exp(-12.0)

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        subsampling_factor: int = 2,
        d_model: int = 640,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 0,
        dropout: float = 0.1,
        **unused,
    ) -> None:
        super().__init__()
        del nhead, dim_feedforward, num_encoder_layers, unused
        if subsampling_factor != 2:
            raise ValueError("The reference TDNN uses subsampling_factor=2")
        if num_decoder_layers != 0:
            raise ValueError("The 5M alignment control is a pure CTC model")

        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.subsampling_factor = 2
        self.hidden_dim = int(d_model)
        self.encoder = nn.ModuleList(
            [
                TdnnBlock(num_features, d_model, 5, 2, 1, dropout),
                TdnnBlock(d_model, d_model, 3, 1, 1, dropout),
                TdnnBlock(d_model, d_model, 3, 1, 1, dropout),
                ResidualFfn(d_model, num_layers=5, dropout=dropout),
            ]
        )
        self.encoder_output_layer = nn.Linear(d_model, num_classes)

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

    def run_encoder(
        self,
        x: Tensor,
        supervisions: Optional[Dict[str, Tensor]] = None,
        warmup: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        del warmup
        if supervisions is None:
            lengths = torch.full(
                (x.size(0),), x.size(1), dtype=torch.long, device=x.device
            )
        else:
            supervision_lengths = supervisions["num_frames"].to(
                device=x.device, dtype=torch.long
            )
            sequence_idx = supervisions.get("sequence_idx")
            if sequence_idx is None:
                lengths = supervision_lengths
            else:
                lengths = torch.full(
                    (x.size(0),), x.size(1), dtype=torch.long, device=x.device
                )
                lengths[
                    sequence_idx.to(device=x.device, dtype=torch.long)
                ] = supervision_lengths

        for layer in self.encoder:
            if isinstance(layer, TdnnBlock):
                x, lengths = layer(x, lengths)
            else:
                x = layer(x)

        padding_mask = (
            torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        )
        return x, lengths, padding_mask

    def forward(
        self,
        x: Tensor,
        supervision: Optional[Dict[str, Tensor]] = None,
        warmup: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        encoded, _, padding_mask = self.run_encoder(x, supervision, warmup)
        log_probs = self.encoder_output_layer(encoded).log_softmax(dim=-1)
        if self.apply_prior_in_forward and float(self.decode_alpha) > 0.0:
            log_probs = apply_label_prior(
                log_probs,
                self.label_prior,
                alpha=float(self.decode_alpha),
                floor=float(self.decode_floor),
                enabled=bool(self.label_prior_ready.item()),
            )
        return log_probs, encoded.transpose(0, 1), padding_mask

    def prior_adjusted_log_probs(
        self,
        log_probs: Tensor,
        alpha: float,
        floor: float,
        enabled: bool = True,
    ) -> Tensor:
        return apply_label_prior(
            log_probs,
            self.label_prior,
            alpha=alpha,
            floor=floor,
            enabled=enabled and bool(self.label_prior_ready.item()),
        )

    @torch.no_grad()
    def reset_label_prior_stats(self) -> None:
        self.label_prior_sum.zero_()
        self.label_prior_total.zero_()

    @torch.no_grad()
    def accumulate_label_prior_stats(
        self, log_probs: Tensor, supervision_segments: Tensor
    ) -> None:
        probs = log_probs.detach().float().exp()
        total_frames = 0
        for sequence_idx, start_frame, num_frames in supervision_segments.tolist():
            start = int(start_frame)
            end = start + int(num_frames)
            if end <= start:
                continue
            segment = probs[int(sequence_idx), start:end]
            self.label_prior_sum.add_(segment.sum(dim=0))
            total_frames += segment.size(0)
        self.label_prior_total.add_(float(total_frames))

    @torch.no_grad()
    def sync_and_update_label_prior(
        self, momentum: float, floor: float
    ) -> Dict[str, float]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.label_prior_sum)
            torch.distributed.all_reduce(self.label_prior_total)

        total = float(self.label_prior_total.item())
        if total > 0.0:
            new_prior = self.label_prior_sum / self.label_prior_total.clamp_min(1.0)
            new_prior = new_prior.clamp_min(float(floor))
            new_prior = new_prior / new_prior.sum()
            if momentum > 0.0:
                new_prior = (
                    float(momentum) * self.label_prior.float()
                    + (1.0 - float(momentum)) * new_prior
                )
                new_prior = new_prior.clamp_min(float(floor))
                new_prior = new_prior / new_prior.sum()
            self.label_prior.copy_(new_prior.to(self.label_prior.dtype))
            self.label_prior_ready.fill_(True)

        prior = self.label_prior.float().clamp_min(float(floor))
        prior = prior / prior.sum()
        stats = {
            "frames": total,
            "ready": float(self.label_prior_ready.item()),
            "blank": prior[0].item(),
            "min": prior.min().item(),
            "max": prior.max().item(),
            "entropy": float(-(prior * prior.log()).sum().item()),
        }
        self.reset_label_prior_stats()
        return stats

    def decoder_forward(self, *args, **kwargs) -> Tensor:
        del args, kwargs
        raise RuntimeError("The 5M TDNN control does not have an AED decoder")
