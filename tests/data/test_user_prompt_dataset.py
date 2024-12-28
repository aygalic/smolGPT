"""Test module for BPE Tokenizer"""

import unittest

from smolgpt.data.user_prompt_dataset import UserPromptDataset


class TestUserPromptDataset(unittest.TestCase):
    """The goal of this test module is to assess that our tokenizer respect some basic nice
    to have properties"""

    def setUp(self):
        """
        Prepare text for training and inference.
        """
        self.basic_prompt: str = "Hello, I am very much a prompt."
        self.long_prompt: str = """Hey I am a very long prompt asking you for a lot of attention.
        This might slip away from your context window and not be taken into account when
        processing. To make sure this is too long I am just going to write one more
        useless sentence."""
        self.block_size = 256

    def tearDown(self):
        """
        This method is called after each test case.
        Use it to clean up any resources created in setUp.
        """

    def test_len(self):
        """Should only yield one query"""
        dataset = UserPromptDataset(self.basic_prompt, self.block_size)
        assert len(dataset) == 1

    def test_short_prompt(self):
        """Yielded prompt should be equal to itself if shorter than context window"""
        dataset = UserPromptDataset(self.basic_prompt, self.block_size)
        prompt = next(iter(dataset))
        assert prompt == self.basic_prompt  # only works if len(prompt)<block_size

    def test_long_prompt(self):
        """Yielded prompt should be truncated to block_size if longer than context window"""
        dataset = UserPromptDataset(self.long_prompt, self.block_size)
        prompt = next(iter(dataset))
        assert len(self.long_prompt) > self.block_size
        assert prompt == self.long_prompt[-self.block_size :]


if __name__ == "__main__":
    unittest.main()
