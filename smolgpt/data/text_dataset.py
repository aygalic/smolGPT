"""Simple dataset class to stream text"""

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Simple dataset to handle a simple stream of tokenized text.

    Parameters
    ----------
    data : list[int]
        Tokenized text
    block_size : int
        Context length
    """

    def __init__(self, data: list[int], block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
