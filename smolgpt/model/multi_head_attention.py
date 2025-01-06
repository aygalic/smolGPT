"""Interface for multi head attention."""

import torch
import torch.nn as nn

from smolgpt.model.attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    """Module to package attention heads into multi head attention

    Parameters
    ----------
    num_heads : int
        Number of heads
    head_size : int
        n_embed // num_heads
    n_embed : int
        model latent dim
    block_size : int
        Context length
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embed: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size, n_embed, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
