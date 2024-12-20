import lightning as L

from torch.utils.data import DataLoader, random_split
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer
from smolgpt.tokenizer.ascii_tokenizer import ASCIITokenizer
from smolgpt.data.instruct_dataset import InstructDataset
from smolgpt.data.user_prompt_dataset import UserPromptDataset
from datasets import load_dataset

class InstructData(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        block_size: int = 256,
        train_val_split: tuple[float, float] = (0.9, 0.1),
        dataset_name: str = "Open-Orca/OpenOrca",
        tokenizer_type: str = None,
        tokenizer_path: str = "tokenizer/vocab/",
        predict_prompt: str = "Once upon a time",  # Default prompt for prediction
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_val_split = train_val_split
        self.dataset_name = dataset_name

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize tokenizer
        if tokenizer_type == "BPE":
            self.tokenizer = BPETokenizer.load(tokenizer_path)
        elif tokenizer_type == "ASCII":
            self.tokenizer = ASCIITokenizer()


    def prepare_data(self):
        """Download dataset and tokenizer if needed"""
        load_dataset(self.dataset_name)
        
    def setup(self, stage: str = None):
        """Set up the datasets for each stage"""
        # Load the dataset
        dataset = load_dataset(self.dataset_name)
        
        if stage == "fit" or stage is None:
            # Split training data
            train_val = dataset["train"].train_test_split(
                test_size=self.train_val_split[1],
                shuffle=True,
                seed=42
            )
            
            # Create training and validation datasets
            self.train_dataset = InstructDataset(
                train_val["train"],
                self.tokenizer,
                self.block_size
            )
            self.val_dataset = InstructDataset(
                train_val["test"],
                self.tokenizer,
                self.block_size
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = InstructDataset(
                dataset["test"] if "test" in dataset else dataset["validation"],
                self.tokenizer,
                self.block_size
            )

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