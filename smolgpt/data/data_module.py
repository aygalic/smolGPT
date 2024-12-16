import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer
from smolgpt.tokenizer.ascii_tokenizer import ASCIITokenizer

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

class GenerationDataset(Dataset):
    def __init__(self, prompt, block_size):
        self.prompt = prompt
        self.block_size = block_size
    
    def __len__(self):
        return 1  # Only one prompt at a time
        
    def __getitem__(self, idx):
        # For generation, we only need the input (no target)
        return self.prompt[-self.block_size:]

class TinyShakespeareData(L.LightningDataModule):
    def __init__(
        self,
        path_to_dir: str = "./data/corpus.txt",
        batch_size: int = 32,
        block_size: int = 256,
        train_val_split: tuple = (0.9, 0.1),
        tokenizer_type: str = None,
        tokenizer_path: str = "tokenizer/vocab/",
        predict_prompt: str = "Once upon a time"  # Default prompt for prediction
    ):
        super().__init__()
        self.save_hyperparameters()
        self.path_to_dir = path_to_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_val_split = train_val_split
        self.predict_prompt = predict_prompt
        
        # Initialize tokenizer
        if tokenizer_type == "BPE":
            self.tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            print("defaulting to ASCII Tokenizer")
            self.tokenizer = ASCIITokenizer()
    
    def prepare_data(self):
        """Called only once and on 1 GPU"""
        pass
    
    def setup(self, stage=None):
        """Called on every GPU"""
        if stage == "predict":
            # For prediction, we only need to prepare the prompt
            prompt_encoded = torch.tensor(self.tokenizer.encode(self.predict_prompt), dtype=torch.long)
            self.predict_dataset = GenerationDataset(prompt_encoded, self.block_size)
            return
            
        # Regular training setup
        with open(self.path_to_dir, "r", encoding="utf-8") as f:
            text = f.read()
        
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        train_len = int(len(data) * self.train_val_split[0])
        val_len = len(data) - train_len
        
        if stage == "fit" or stage is None:
            dataset_train, dataset_val = random_split(
                TextDataset(data, self.block_size),
                [train_len - self.block_size, val_len],
                generator=torch.Generator().manual_seed(42),
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
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,  # We generate one sequence at a time
            shuffle=False,
        )