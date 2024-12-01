import torch

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