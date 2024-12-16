import torch.nn as nn
import torch
from smolgpt.model.block import Block
from torch.nn import functional as F
import lightning as L


class Transformer(L.LightningModule):
    def __init__(
            self,
            #vocab_size: int,
            n_embed: int,
            block_size: int,
            vocab_size: int,
            n_heads: int,
            n_layer: int,
            dropout: float,
            learning_rate: float = 3e-4,
            device_type: str = "mps"  # Changed from device to device_type
            ):
        super().__init__()
        self.save_hyperparameters()

        # reference hyperparameters through self.hparams
        self.n_heads = self.hparams.n_heads
        self.n_layer = self.hparams.n_layer
        self.n_embed = self.hparams.n_embed
        self.dropout = self.hparams.dropout
        self.block_size = self.hparams.block_size
        self.device_type = self.hparams.device_type
        self.learning_rate = self.hparams.learning_rate
        self.vocab_size = self.hparams.vocab_size

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)
        self.blocks = nn.Sequential(*[Block(self.n_embed, self.n_heads, self.block_size, self.dropout) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embed)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size) # Language Model


    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C) C = n_embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device_type)) # (T, C) C = n_embed
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # apply self attention (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        In Lightning's predict_step, batch can be a tuple where we pack additional arguments
        """
        if isinstance(batch, tuple):
            idx, max_new_tokens = batch
        else:
            idx = batch
            max_new_tokens = 100  # default value if not specified
            
        return self.generate(idx, max_new_tokens)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr = self.learning_rate)

    def generate(self, idx, max_new_tokens):
        # idx shape is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            # get preds:
            logits = self(idx_cond)
            # focus on last time step:
            logits = logits[:, -1, :] # (B, C)
            # softmax for probs
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample from dist:
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the sampled token to the running sequence 
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
    
