"""draft for implementing tokenization"""
from collections import Counter
text = """the full recipe that defines how your nn.Modules interact.

    The training_step defines how the nn.Modules interact together.

    In the configure_optimizers define the optimizer(s) for your models.

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder#😎"""

tokens = text.encode("utf-8")
tokens = list(map(int, tokens))
print(text)
print("-----")
print(len(text))
print("-----")


print(tokens)
print("-----")
print(len(tokens))
print("-----")

# byte pair encoding

def get_stats(ids):
    return Counter(zip(ids, ids[1:]))

stats = get_stats(tokens)
print(stats)
print(sorted(((v,k) for k,v in stats.items()), reverse=True) )

top_pair = max(stats, key = stats.get)
print(top_pair)

def merge(ids, pair, idx):
    newids = []
    i=0
    while i<len(ids):
        if i<len(ids)-1 and ids[i] == pair[0] and  ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

print(merge([5,6,6,7,9,1], (6,7), 99))

tokens2 = merge(tokens, top_pair, 256)
print(tokens2)
print(len(tokens2))

# -----

vocab_size = 276
num_merges = vocab_size - 256 # original number of tokens
ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key = stats.get)
    idx = 256 + i
    print(f"merged {pair=} into new token {idx=}")
    ids= merge(ids, pair, idx)
    merges[pair]=idx

print(ids)
print(len(tokens))
print(len(ids))
print(f"compression ration = {len(tokens)/len(ids)}")

# ---- decoding

reversed_map = {v:k for k,v in merges.items()}
print(reversed_map)



def decode(ids):
    _decoded_seq = []
    for token in ids:
        if token in reversed_map.keys():
            new_tokens = reversed_map[token]
            out = decode(new_tokens)
        else :
            out = [token]
        _decoded_seq += out
    return _decoded_seq

decoded_sec = decode(ids)
print(decoded_sec)
print(len(decoded_sec))

breakpoint()

