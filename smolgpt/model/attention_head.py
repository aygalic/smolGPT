import torch.nn as nn
import torch
from torch.nn import functional as F

class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, 16)
        q = self.query(x) # (B, T, 16)
        v = self.value(x) # (B, T, 16)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T), ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0 , float("-inf"))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        v = self.value(x)
        out  = wei @ v
        return out
