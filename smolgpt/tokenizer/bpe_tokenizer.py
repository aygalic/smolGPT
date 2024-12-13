from collections import Counter
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size:int):
        assert vocab_size >= 255
        self.vocab_size=vocab_size
        self.merges = {}

    def _get_stats(self, ids: list[int]) -> dict[tuple[int, int], int]:
        return Counter(zip(ids, ids[1:]))

    def _merge(self, ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

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

        #print(ids)
        #print(len(tokens))
        #print(len(ids))
        print(f"compression ration = {len(tokens)/len(ids)}")

        # ---- decoding
        self.reversed_map = {v: k for k, v in self.merges.items()}
        print(self.reversed_map)

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
