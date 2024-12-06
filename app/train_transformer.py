"""Draft for transformers"""
from lightning import Trainer
import torch
from smolgpt.model.transformer import Transformer
from smolgpt.data.data_module import TinyShakespeareData

DEVICE = "mps"
torch.manual_seed(123)
block_size = 256
batch_size = 64
learning_date = 3e-4
max_iter = 5000
eval_iter= 200
eval_interval = 300
n_heads = 2
n_embed = n_heads*32
n_layer = 2
dropout = 0.2


data_module = TinyShakespeareData(
    path_to_dir="./data/corpus.txt",
    batch_size=64,
    block_size=256
)
data_module.setup()
# Access vocab size for model initialization
vocab_size = data_module.vocab_size

model = Transformer(vocab_size, n_embed, block_size, n_heads, n_layer, dropout)

trainer = Trainer(accelerator="mps", max_steps=max_iter)
trainer.fit(model, data_module)

m = model.to(DEVICE)
idx = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
output = data_module.decode(m.generate(idx, max_new_tokens=1000)[0].tolist())
print("".join(output))
