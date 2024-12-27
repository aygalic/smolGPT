"""Data Module for instruct Dataset"""
import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from smolgpt.data.instruct_dataset import InstructDataset
from smolgpt.data.user_prompt_dataset import UserPromptDataset
from smolgpt.tokenizer.ascii_tokenizer import ASCIITokenizer
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer


class InstructData(L.LightningDataModule):
    """Data Module for the instruct dataset"""
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
        """aaaa

        Parameters
        ----------
        batch_size : int, optional
            Batch Size, by default 32
        block_size : int, optional
            Context length, by default 256
        train_val_split : tuple[float, float], optional
            Train set ratio, val set ratio, must sum to 1, by default (0.9, 0.1)
        dataset_name : str, optional
            Name of the dataset, by default "Open-Orca/OpenOrca"
        tokenizer_type : str, optional
            Type of tokenizer, pick between ASCII and BPE, by default None
        tokenizer_path : str, optional
            Path for tokenizer file, by default "tokenizer/vocab/"
        predict_prompt : str, optional
            Prompt to use when generating new tokens, by default "Once upon a time"
        """
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.block_size = block_size
        self.train_val_split = train_val_split
        self.dataset_name = dataset_name

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.predict_prompt = predict_prompt

        # Initialize tokenizer
        if tokenizer_type == "BPE":
            self.tokenizer = BPETokenizer.load(tokenizer_path)
        elif tokenizer_type == "ASCII":
            self.tokenizer = ASCIITokenizer()

    def prepare_data(self):
        """Download dataset
        FIXME The tokenization should take place here
        """
        self.full_dataset = load_dataset(self.dataset_name)

    def setup(self, stage: str = None):
        """Set up the datasets for each stage"""
        # Load the dataset
        dataset = self.full_dataset

        if stage == "predict":
            # For prediction, we only need to prepare the prompt
            prompt = f"System: You are an AI agent that provide useful help.\nQuestion: {self.predict_prompt}\n"
            full_prompt = prompt + f"Response:"
            prompt_encoded = torch.tensor(
                self.tokenizer.encode(full_prompt), dtype=torch.long
            )
            self.predict_dataset = UserPromptDataset(prompt_encoded, self.block_size)
            return

        if stage == "fit" or stage is None:
            # Split training data
            train_val = dataset["train"].train_test_split(
                test_size=self.train_val_split[1], shuffle=True, seed=42
            )

            # Create training and validation datasets
            self.train_dataset = InstructDataset(
                train_val["train"], self.tokenizer, self.block_size
            )
            self.val_dataset = InstructDataset(
                train_val["test"], self.tokenizer, self.block_size
            )

        if stage == "test" or stage is None:
            self.test_dataset = InstructDataset(
                dataset["test"] if "test" in dataset else dataset["validation"],
                self.tokenizer,
                self.block_size,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,  # We generate one sequence at a time
            shuffle=False,
        )
