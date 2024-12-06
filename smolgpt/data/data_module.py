import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

class TinyShakespeareData(L.LightningDataModule):
    def __init__(
        self,
        path_to_dir="./data/corpus.txt",
        batch_size=32,
        block_size=256,
        train_val_split=(0.9, 0.1)
    ):
        super().__init__()
        self.path_to_dir = path_to_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_val_split = train_val_split
        self.stoi = None
        self.itos = None
        self.vocab_size = None
        
    def encode(self, x):
        """Convert string to list of integers"""
        return [self.stoi[c] for c in x]
    
    def decode(self, x):
        """Convert list of integers back to string"""
        return [self.itos[i] for i in x]
    
    def prepare_data(self):
        """Called only once and on 1 GPU"""
        # You can download data here if needed
        pass
    
    def setup(self, stage=None):
        """Called on every GPU"""
        # Read and process the text file
        with open(self.path_to_dir, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Create vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {i: s for i, s in enumerate(chars)}
        
        # Encode the full text
        data = torch.tensor(self.encode(text), dtype=torch.long)
        
        # Create train/val split
        train_len = int(len(data) * self.train_val_split[0])
        val_len = len(data) - train_len
        
        # Create datasets
        if stage == "fit" or stage is None:
            dataset_train, dataset_val = random_split(
                TextDataset(data, self.block_size),
                [train_len - self.block_size, val_len],
                generator=torch.Generator().manual_seed(42)
            )
            self.train_dataset = dataset_train
            self.val_dataset = dataset_val
            
        if stage == "test" or stage is None:
            self.test_dataset = TextDataset(data[-val_len:], self.block_size)
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )