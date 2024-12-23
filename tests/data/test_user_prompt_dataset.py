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
        self.block_size = 256

    def tearDown(self):
        """
        This method is called after each test case.
        Use it to clean up any resources created in setUp.
        """

    def testLen(self):
        dataset = UserPromptDataset(self.basic_prompt, self.block_size)
        assert len(dataset)==1

    def testPrompt(self):
        dataset = UserPromptDataset(self.basic_prompt, self.block_size)
        prompt = next(iter(dataset))
        assert prompt==self.basic_prompt # only works if len(prompt)<block_size



if __name__ == '__main__':
    unittest.main()
