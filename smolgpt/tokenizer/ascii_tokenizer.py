from smolgpt.tokenizer.tokenizer import TokenizerABC

class ASCIITokenizer(TokenizerABC):
    def __init__(self):
        #FIXME : self init using tiny shakespeer, this should NOT be a thing
        path_to_dir="./data/corpus.txt"
        with open(path_to_dir, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {i: s for i, s in enumerate(chars)}


    def encode(self, x):
        """Convert string to list of integers"""
        return [self.stoi[c] for c in x]
    
    def decode(self, x):
        """Convert list of integers back to string"""
        return [self.itos[i] for i in x]
