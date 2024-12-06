import lightning as L
import torch
from torch.utils.data import random_split, DataLoader
class TinyShakespeareData(L.LightningDataModule):

    def __init__(self, path_to_dir = "./data/corpus.txt", batch_size = 32):
        super().__init__()
        self.path_to_dir = path_to_dir
        self.batch_size = batch_size


        self.stoi 
        self.itos 


    def encode(self, x):
        return [self.stoi[c] for c in x]

    def decode(self, x):
        return [self.itos[c] for c in x]

    def setup(self, stage: str):
        with open(self.path_to_dir, "r", encoding="utf-8") as f:
            text = f.read()


        chars = sorted(list(set(text)))

        self.stoi = {s:i for (i,s) in enumerate(chars)}
        self.itos = {i:s for (i,s) in enumerate(chars)} # reverse map


        vocab_size = len(chars)


        # tokenize data
        data = torch.tensor(self.encode(text), dtype = torch.long )

        # train/test split:
        self.tiny_shakespeare_train, self.tiny_shakespeare_test = random_split(
            data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.tiny_shakespeare_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.tiny_shakespeare_test, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.tiny_shakespeare_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.tiny_shakespeare_test, batch_size=self.batch_size)