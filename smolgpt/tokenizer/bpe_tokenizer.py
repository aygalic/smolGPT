import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from smolgpt.tokenizer.tokenizer import TokenizerABC

class BPETokenizer(TokenizerABC):
    def __init__(self, vocab_size: int):
        assert vocab_size >= 255
        self.vocab_size = vocab_size
        self.merges = {}
        self.reversed_map = {}

    def fit(self, text: str):
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        num_merges = self.vocab_size - 256  # original number of tokens
        ids = list(tokens)
        for i in tqdm(range(num_merges)):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            self.merges[pair] = idx
        print(f"compression ration = {len(tokens)/len(ids)}")
        self.reversed_map = {v: k for k, v in self.merges.items()}

    def encode(self, text: str) -> list[int]:
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        ids = list(tokens)

        while True:
            encoded_seq_ = []
            i = 0
            found_pairs = False
            while i < len(ids):
                if i == len(ids) - 1:
                    encoded_seq_.append(ids[i])
                    break
                pair = (ids[i], ids[i + 1])
                if pair in self.merges.keys():
                    tok = self.merges[pair]
                    encoded_seq_.append(tok)
                    found_pairs = True
                    i += 2
                else:
                    encoded_seq_.append(ids[i])
                    i += 1
            ids = encoded_seq_

            if not found_pairs:
                return ids

    def decode(self, ids: list[int]) -> str:
        def decode_recursive(ids):
            _decoded_seq = []
            for token in ids:
                if token in self.reversed_map.keys():
                    pair = self.reversed_map[token]
                    _decoded_seq += decode_recursive(list(pair))
                else:
                    _decoded_seq += [token]
            return _decoded_seq

        out = decode_recursive(ids)
        out = bytes(out)
        return out.decode("utf8", errors="replace")

    @staticmethod
    def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def save(self, save_dir: str):
        """Save the tokenizer's merge rules to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        serializable_merges = {
            f"{pair[0]},{pair[1]}": idx for pair, idx in self.merges.items()
        }
        with open(save_path / "merges.json", "w", encoding="utf-8") as f:
            json.dump(
                {"vocab_size": self.vocab_size, "merges": serializable_merges},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, save_dir: str) -> "BPETokenizer":
        """Load a tokenizer from saved merge rules."""
        save_path = Path(save_dir)
        with open(save_path / "merges.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(data["vocab_size"])
        tokenizer.merges = {
            tuple(map(int, k.split(","))): v for k, v in data["merges"].items()
        }
        tokenizer.reversed_map = {v: k for k, v in tokenizer.merges.items()}
        return tokenizer
