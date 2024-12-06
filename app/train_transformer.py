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


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k]= loss.item()
        out[split] = losses.mean()
    model.train()
    return out

with open("./data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"length of the dataset in characters : {len(text)}")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vacabulary size : {vocab_size=}")


# tokenizer
stoi = {s:i for (i,s) in enumerate(chars)}
itos = {i:s for (i,s) in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : [itos[i] for i in l]


# tokenize data
data = torch.tensor(encode(text), dtype = torch.long )

# train/test split:
n = int(len(data)*0.9)
train_data = data[:n]
test_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

data_module = TinyShakespeareData(
    path_to_dir="./data/corpus.txt",
    batch_size=64,
    block_size=256
)
data_module.setup()
# Access vocab size for model initialization
vocab_size = data_module.vocab_size



model = Transformer(vocab_size, n_embed, block_size, n_heads, n_layer, dropout)

trainer = Trainer()
trainer.fit(model, data_module)

m = model.to(DEVICE)

idx = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
output = decode(m.generate(idx, max_new_tokens=100)[0].tolist())
print("".join(output))

# trianing the model
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_date)

for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}, train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    #print(loss.item())

context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
output = decode(m.generate(idx, max_new_tokens=100)[0].tolist())
print("".join(output))

