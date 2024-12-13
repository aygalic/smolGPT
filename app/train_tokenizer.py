"""draft for implementing tokenization"""
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer

from datasets import load_dataset

ds = load_dataset("rahular/simple-wikipedia")



text = "".join(ds["train"][:10000]["text"])


breakpoint()


tokenizer = BPETokenizer()
tokenizer.fit(text)

print(f'{tokenizer.encode(text)=}')
out = tokenizer.decode(tokenizer.encode(text))

print(out)
print("compression ratio =",len(tokenizer.decode(tokenizer.encode(text)))/ len(tokenizer.encode(text)) )


# special tokens
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""