"""draft for implementing tokenization"""

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
