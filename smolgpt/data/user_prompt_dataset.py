"""This module s used to feed a user generated string into the model through the Dataset
API"""

from torch.utils.data import Dataset


class UserPromptDataset(Dataset):
    """Make a dataset of a single prompt

    Parameters
    ----------
    prompt : list[int]
        Prompt to be fed (tokenized)
    block_size : int
        Context window size
    """

    def __init__(self, prompt: list[int], block_size: int):
        self.prompt = prompt
        self.block_size = block_size

    def __len__(self):
        return 1  # Only one prompt at a time

    def __getitem__(self, idx):
        # For generation, we only need the input (no target)
        return self.prompt[-self.block_size :]
