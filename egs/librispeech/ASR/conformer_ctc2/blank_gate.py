import math

import torch
import torch.nn as nn
from torch import Tensor


class BlankGateHead(nn.Module):
    """
    Posterior  q_psi(z_b^t = 1 | X, Y)

    VarCTC-style (ECCV 2020): mean-pool label embeddings over Y, then
    Hadamard product with projected encoder hidden states.  Batch-level
    computation — no per-utterance loop, no cross-attention weights to
    initialise from scratch.

    init_blank_prob: desired sigmoid(bias) at init (non-blank probability).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        label_embed_dim: int = 256,
        init_blank_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(vocab_size, label_embed_dim, padding_idx=0)
        self.x_proj = nn.Linear(d_model, label_embed_dim)
        self.out = nn.Linear(label_embed_dim, 1)

        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.x_proj.bias, 0.0)

        init_blank_prob = float(max(1e-4, min(1.0 - 1e-4, init_blank_prob)))
        nn.init.constant_(
            self.out.bias, math.log(init_blank_prob / (1.0 - init_blank_prob))
        )

    def forward(
        self,
        encoder_out: Tensor,    # [N, T, d_model]
        targets: Tensor,        # [N, U_max]  padded BPE token ids (0 = padding)
        target_lengths: Tensor, # [N]  valid token counts per utterance
    ) -> Tensor:
        """
        Returns alpha [N, T]  sigmoid non-blank probabilities.
        Padding frame positions are NOT explicitly zeroed — the caller applies
        valid_mask if the loss requires it (OT/KL already skip padding via
        per-utterance slicing [:L]).
        """
        emb = self.label_embedding(targets)                       # [N, U_max, D]
        U_max = targets.size(1)
        tok_idx = torch.arange(U_max, device=targets.device).unsqueeze(0)
        tok_mask = (tok_idx < target_lengths.unsqueeze(1)).unsqueeze(-1).to(emb.dtype)
        # mean-pool over valid tokens only
        y_bar = (emb * tok_mask).sum(1) / target_lengths.clamp_min(1).unsqueeze(1).to(emb.dtype)
        # [N, D]

        x_proj = self.x_proj(encoder_out)                         # [N, T, D]
        gated = x_proj * y_bar.unsqueeze(1)                       # [N, T, D]  Hadamard
        return torch.sigmoid(self.out(gated).squeeze(-1))         # [N, T]


class BlankPriorHead(nn.Module):
    """
    Prior  p_beta(z_b^t = 1 | X)

    Linear + sigmoid on encoder hidden states.  Used at inference time
    (requires only X, no transcript Y).

    init_blank_prob: desired sigmoid(bias) at init (non-blank probability).
    """

    def __init__(self, d_model: int, init_blank_prob: float = 0.5) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

        init_blank_prob = float(max(1e-4, min(1.0 - 1e-4, init_blank_prob)))
        nn.init.constant_(
            self.linear.bias, math.log(init_blank_prob / (1.0 - init_blank_prob))
        )

    def forward(self, encoder_out: Tensor, encoder_out_lens: Tensor) -> Tensor:
        """
        encoder_out:      [N, T, d_model]
        encoder_out_lens: [N]  valid frame counts
        Returns: [N, T]   sigmoid probs; padding positions are zeroed out-of-place.
        """
        logit = self.linear(encoder_out).squeeze(-1)    # [N, T]
        alpha_tilde = torch.sigmoid(logit)
        T = alpha_tilde.size(1)
        frame_idx = torch.arange(T, device=alpha_tilde.device).unsqueeze(0)
        valid_mask = frame_idx < encoder_out_lens.unsqueeze(1)
        return alpha_tilde * valid_mask.to(alpha_tilde.dtype)
