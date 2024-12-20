from torch.utils.data import Dataset
import torch
import torch.nn.functional as F



class InstructDataset(Dataset):

    def __init__(self, data, tokenizer, block_size: int):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Combine system prompt, question and response
        prompt = f"System: {item['system_prompt']}\nQuestion: {item['question']}\n"
        full_text = prompt + f"Response: {item['response']}"
        
        # Tokenize both prompt and full text
        prompt_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
        full_ids = torch.tensor(self.tokenizer.encode(full_text), dtype=torch.long)
        
        # Pad or truncate to block_size
        if len(full_ids) > self.block_size:
            full_ids = full_ids[:self.block_size]
        else:
            full_ids = F.pad(full_ids, (0, self.block_size - len(full_ids)), value=0)
        
        # For targets, we want -100 for prompt tokens (they won't contribute to loss)
        targets = full_ids.clone()
        targets[:len(prompt_ids)] = -100
        
        return full_ids, targets