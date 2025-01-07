"""Define base architecture for tokenizer"""

from abc import ABC


class TokenizerABC(ABC):
    """Abstract base class for tokenizers"""

    def encode(self, x: str) -> list[int]:
        """Encode a given string

        Parameters
        ----------
        x : str
            String to encode

        Returns
        -------
        list[int]
            list of tokens
        """

    def decode(self, x: list[int]) -> str:
        """Decode a list of tokens

        Parameters
        ----------
        x : list[int]
            List of tokens to decode

        Returns
        -------
        str
            decoded string
        """
