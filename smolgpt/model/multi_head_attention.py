from smolgpt.model.attention_head import AttentionHead
import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
