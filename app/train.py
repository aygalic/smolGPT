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

