import torch
import torch.nn as nn
from torch.nn import functional as F


with open("./data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"length of the dataset in characters : {len(text)}")
chars = sorted(list(set(text)))
vacab_size = len(chars)
print("".join(chars))
print(f"Vacabulary size : {vacab_size=}")


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


torch.manual_seed(123)
batchs_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batchs_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)

print("targets:")
print(yb.shape)
print(yb)

print("------")

for b in range(batchs_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"When {context=}, {target=}")

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) # (B, T, C)
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
    
m = BigramLanguageModel(vacab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))