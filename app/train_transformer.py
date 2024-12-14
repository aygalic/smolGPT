"""Train transformer with checkpointing"""

import json

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from smolgpt.data.data_module import TinyShakespeareData
from smolgpt.model.transformer import Transformer
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = "mps"
torch.manual_seed(123)
block_size = 256
batch_size = 64
max_iter = 50

n_heads = 4
n_embed = n_heads * 32
n_layer = 2
dropout = 0.2


# Initialize tokenizer and data module
tokenizer = BPETokenizer.load("tokenizer/vocab/")
data_module = TinyShakespeareData(
    path_to_dir="./data/corpus.txt",
    batch_size=batch_size,
    block_size=block_size,
    tokenizer=tokenizer,
)
data_module.setup()
vocab_size = data_module.tokenizer.vocab_size

# Initialize model
model = Transformer(vocab_size, n_embed, block_size, n_heads, n_layer, dropout)


# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="transformer-{step}",
    save_top_k=3,
    verbose=True,
    monitor="train_loss",
    mode="min",
)

# Initialize trainer with checkpoint callback
trainer = Trainer(
    accelerator="mps", max_steps=max_iter, callbacks=[checkpoint_callback]
)

# Train model
trainer.fit(model, data_module)

# Save the tokenizer configuration along with the model
model_info = {
    "block_size": block_size,
    "n_heads": n_heads,
    "n_embed": n_embed,
    "n_layer": n_layer,
    "dropout": dropout,
    "vocab_size": vocab_size,
}

with open("checkpoints/model_config.json", "w") as f:
    json.dump(model_info, f)
