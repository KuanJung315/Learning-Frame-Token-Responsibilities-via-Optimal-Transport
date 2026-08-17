import math

import torch
import torch.nn as nn
from torch import Tensor


class BlankGateHeadV2(nn.Module):
    """
    Cross-attention posterior gate q_psi(z_b^t = 1 | X, Y).

    Each encoder frame attends over the utterance label embeddings, producing a
    per-frame label context.  The Hadamard interaction keeps the VarCTC-style
    multiplicative gate, but avoids the v1 mean-pool collapse where every frame
    saw the same transcript summary.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        d_attn: int = 256,
        init_blank_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(vocab_size, d_attn, padding_idx=0)
        self.q_proj = nn.Linear(d_model, d_attn, bias=False)
        self.out = nn.Linear(d_attn, 1)
        self.scale = d_attn**-0.5

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out.weight)
        init_blank_prob = float(max(1e-4, min(1.0 - 1e-4, init_blank_prob)))
        nn.init.constant_(
            self.out.bias,
            math.log(init_blank_prob / (1.0 - init_blank_prob)),
        )

    def forward(
        self,
        encoder_out: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        encoder_out_lens: Tensor,
    ) -> Tensor:
        """
        Args:
          encoder_out:
            Batch-first encoder hidden states, [N, T, d_model].
          targets:
            Padded BPE token IDs, [N, U_max], where 0 is padding/blank.
          target_lengths:
            Valid token counts per utterance, [N].
          encoder_out_lens:
            Valid encoder frame counts per utterance, [N].

        Returns:
          Per-frame non-blank probabilities alpha, [N, T].  Padding frames are
          zeroed so OT/KL metrics cannot accidentally consume padded mass.
        """
        N, T, _ = encoder_out.shape
        U_max = targets.size(1)

        if U_max == 0:
            return encoder_out.new_zeros((N, T))

        emb = self.label_embedding(targets)
        q = self.q_proj(encoder_out)
        # Autocast may compute the attention logits in fp16.  Do the masking and
        # softmax in fp32 so the padding sentinel cannot overflow half tensors.
        attn_logits = (torch.bmm(q, emb.transpose(1, 2)) * self.scale).float()

        tok_mask = (
            torch.arange(U_max, device=targets.device).unsqueeze(0)
            >= target_lengths.unsqueeze(1)
        )
        attn_logits = attn_logits.masked_fill(tok_mask.unsqueeze(1), -1.0e9)
        attn_weights = torch.softmax(attn_logits, dim=-1).to(emb.dtype)

        y_t = torch.bmm(attn_weights, emb)
        alpha = torch.sigmoid(self.out(q * y_t).squeeze(-1))

        frame_idx = torch.arange(T, device=encoder_out.device).unsqueeze(0)
        valid_mask = frame_idx < encoder_out_lens.unsqueeze(1)
        return alpha * valid_mask.to(alpha.dtype)


class BlankPriorHeadV2(nn.Module):
    """
    Prior p_beta(z_b^t = 1 | X).

    The prior is intentionally X-only so it can replace the posterior gate at
    inference time, where transcripts are unavailable.
    """

    def __init__(self, d_model: int, init_blank_prob: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

        init_blank_prob = float(max(1e-4, min(1.0 - 1e-4, init_blank_prob)))
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(
            self.linear.bias,
            math.log(init_blank_prob / (1.0 - init_blank_prob)),
        )

    def forward(self, encoder_out: Tensor, encoder_out_lens: Tensor) -> Tensor:
        logit = self.linear(encoder_out).squeeze(-1)
        alpha_tilde = torch.sigmoid(logit)

        T = alpha_tilde.size(1)
        frame_idx = torch.arange(T, device=alpha_tilde.device).unsqueeze(0)
        valid_mask = frame_idx < encoder_out_lens.unsqueeze(1)
        return alpha_tilde * valid_mask.to(alpha_tilde.dtype)
