"""Data module designed for the pre-training task using mask learning modelling"""

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from smolgpt.data.text_dataset import TextDataset
from smolgpt.data.user_prompt_dataset import UserPromptDataset
from smolgpt.tokenizer.ascii_tokenizer import ASCIITokenizer
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer


class TinyShakespeareData(L.LightningDataModule):
    """Data module based on the tiny Shakespeare dataset.

    Parameters
    ----------
    path_to_dir : str, optional
        Path where the data is saved as a txt file, by default "./data/corpus.txt"
    batch_size : int, optional
        Batch size, by default 32
    block_size : int, optional
        Context length, by default 256
    train_val_split : tuple, optional
        Train set and validation set portion, must sum to 1, by default (0.9, 0.1)
    tokenizer_type : str, optional
        Type of tokenizer to use, between ASCII and BPE, by default None
    tokenizer_path : str, optional
        Path for tokenizer data (used for BPE only), by default "tokenizer/vocab/"
    predict_prompt : str, optional
        A prompt to use when testing the output of the model, by default "Once upon a time"
    """

    def __init__(
        self,
        path_to_dir: str = "./data/corpus.txt",
        batch_size: int = 32,
        block_size: int = 256,
        train_val_split: tuple = (0.9, 0.1),
        tokenizer_type: str = None,
        tokenizer_path: str = "tokenizer/vocab/",
        predict_prompt: str = "Once upon a time",  # Default prompt for prediction
    ):
        super().__init__()
        assert tokenizer_type in ["ASCII", "BPE"]
        self.save_hyperparameters()
        self.path_to_dir = path_to_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_val_split = train_val_split
        self.predict_prompt = predict_prompt

        # Initialize tokenizer
        if tokenizer_type == "BPE":
            self.tokenizer = BPETokenizer.load(tokenizer_path)
        elif tokenizer_type == "ASCII":
            self.tokenizer = ASCIITokenizer()

    def prepare_data(self):
        """Called only once and on 1 GPU"""
        pass

    def setup(self, stage=None):
        """Called on every GPU"""
        if stage == "predict":
            # For prediction, we only need to prepare the prompt
            prompt_encoded = torch.tensor(
                self.tokenizer.encode(self.predict_prompt), dtype=torch.long
            )
            self.predict_dataset = UserPromptDataset(prompt_encoded, self.block_size)
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
