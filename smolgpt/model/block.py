"""Layer in transformer architecture."""
import torch.nn as nn
from torch import Tensor

from smolgpt.model.feed_forward import FeedForward
from smolgpt.model.multi_head_attention import MultiHeadAttention


class Block(nn.Module):
    """This is considered as a Layer in a classic transformer.

    Parameters
    ----------
    n_embed : int
        Model latent dimension.
    num_heads : int
        Number of attention heads.
    block_size : int
        Context length.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_embed: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embed, block_size, dropout)
        self.ffw = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x
