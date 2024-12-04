from smolgpt.model.feed_forward import FeedForward
from smolgpt.model.multi_head_attention import MultiHeadAttention

import torch.nn as nn

class Block(nn.Module):
    def __init__(self, n_embed, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embed//num_heads 
        self.sa = MultiHeadAttention(num_heads, head_size, n_embed, block_size, dropout)
        self.ffw = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x
