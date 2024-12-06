import torch.nn as nn
import torch
from smolgpt.model.block import Block
from torch.nn import functional as F
import lightning as L

DEVICE = "mps"
learning_rate = 3e-4

class Transformer(L.LightningModule):
    def __init__(self, vocab_size, n_embed, block_size, n_heads, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # Language Model

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C) C = n_embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C) C = n_embed
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # apply self attention (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        xb, yb = batch
        logits, loss = self(xb, yb)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr = learning_rate)

    def generate(self, idx, max_new_tokens):
        # idx shape is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            # get preds:
            logits, loss = self(idx_cond)
            # focus on last time step:
            logits = logits[:, -1, :] # (B, C)
            # softmax for probs
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample from dist:
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the sampled token to the running sequence 
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
    
