"""Draft for transformers"""
import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = "mps"
torch.manual_seed(123)
block_size = 8
batch_size = 32
learning_date = 1e-3
max_iter = 5000
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

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, 16)
        q = self.query(x) # (B, T, 16)
        v = self.value(x) # (B, T, 16)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T), ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0 , float("-inf"))
        wei = F.softmax(wei, dim = -1)
        v = self.value(x)
        out  = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out
    
class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed//num_heads 
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffw = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffw(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            Block(n_embed, num_heads=4),
            Block(n_embed, num_heads=4),
            Block(n_embed, num_heads=4),
        )

        self.lm_head = nn.Linear(n_embed, vocab_size) # Language Model


    def forward(self, idx, targets = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C) C = n_embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C) C = n_embed
        x = tok_emb + pos_emb

        x = self.blocks(x) # apply self attention (B, T, C)

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
            idx_cond = idx[:, -block_size:]
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
    
model = BigramLanguageModel()
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

