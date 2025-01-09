from smolgpt.tokenizer.tokenizer import TokenizerABC

class ASCIITokenizer(TokenizerABC):
    """ASCII tokenizer based on tiny shakespeare data"""
    def __init__(self):
        text = "!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        chars = list(set(text))
        chars += [" ", "\n"]
        chars = sorted(chars)

        self.vocab_size = len(chars)
        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {i: s for i, s in enumerate(chars)}


    def encode(self, x: str) -> list[int]:
        """Convert string to list of integers"""
        return [self.stoi[c] for c in x]
    
    def decode(self, x: list[int]) -> str:
        """Convert list of integers back to string"""
        return [self.itos[i] for i in x]
