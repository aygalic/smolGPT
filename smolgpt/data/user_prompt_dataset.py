from torch.utils.data import Dataset


class UserPromptDataset(Dataset):
    def __init__(self, prompt, block_size):
        self.prompt = prompt
        self.block_size = block_size

    def __len__(self):
        return 1  # Only one prompt at a time

    def __getitem__(self, idx):
        # For generation, we only need the input (no target)
        return self.prompt[-self.block_size :]
