"""draft for implementing tokenization"""
from smolgpt.tokenizer.bpe_tokenize import PBETokenize

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

tokenizer = PBETokenize()
tokenizer.fit(text)

print(f'{tokenizer.encode(text)=}')
out = tokenizer.decode(tokenizer.encode(text))

print(out)

print("compression ratio =",len(tokenizer.decode(tokenizer.encode(text)))/ len(tokenizer.encode(text)) )


# special tokens
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""