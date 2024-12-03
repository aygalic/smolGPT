"""Draft for transformers"""
import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = "mps"
torch.manual_seed(123)
block_size = 8
batch_size = 32
learning_date = 1e-2
max_iter = 3000
eval_iter= 200
eval_interval = 300
n_embed = 32


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
print("".join(chars))
print(f"Vacabulary size : {vocab_size=}")


# tokenizer
stoi = {s:i for (i,s) in enumerate(chars)}
itos = {i:s for (i,s) in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : [itos[i] for i in l]

print(encode("test"))
print(decode(encode("test")))

# tokenize data
data = torch.tensor(encode(text), dtype = torch.long )
print(data.shape, data.dtype)
print(data[:10])

# train/test split:
n = int(len(data)*0.9)
train_data = data[:n]
test_data = data[n:]

# context_length / block_size
context_length = 8
print(f"first block is {train_data[:context_length+1]}")

for i in range(context_length+1):
    print(f"context is {train_data[:i]}, target is {train_data[i]}")



def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)

print("targets:")
print(yb.shape)
print(yb)

print("------")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"When {context=}, {target=}")

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # Language Model


    def forward(self, idx, targets = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C) C = n_embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C) C = n_embed
        x = tok_emb + pos_emb
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx shape is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get preds:
            logits, loss = self(idx)
            # focus on last time step:
            logits = logits[:, -1, :] # (B, C)
            # softmax for probs
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample from dist:
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the sampled token to the running sequence 
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(DEVICE)

logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
idx = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

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


# Mathematical trick for self attention:

B, T, C = 4,8,2
x = torch.randn(B, T, C)
print(x.shape)

tril = torch.tril(torch.ones(T,T))
xbow = torch.zeros((B, T, C))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim = -1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)