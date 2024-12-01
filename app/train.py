
with open("./data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"length of the dataset in characters : {len(text)}")
chars = sorted(list(set(text)))
vacab_size = len(chars)
print("".join(chars))
print(f"Vacabulary size : {vacab_size=}")


# encoder/decoder
stoi = {s:i for (i,s) in enumerate(chars)}
itos = {i:s for (i,s) in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : [itos[i] for i in l]

print(encode("test"))
print(decode(encode("test")))