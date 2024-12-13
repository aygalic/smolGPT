"""Test module for BPE Tokenizer"""
import unittest
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer

class TestClassBPETokenizer(unittest.TestCase):
    """The goal of this test module is to assess that our tokenizer respect some basic nice
    to have properties"""
    def setUp(self):
        """
        Prepare text for training and inference.
        """
        self.training_text = """the full recipe that defines how your nn.Modules
    interact.

    The training_step defines how the nn.Modules interact together.

    In the configure_optimizers define the optimizer(s) for your models.

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder#😎"""

        self.test_text = """This template includes the essential components for writing
unit tests in Python:

A test class that inherits from unittest.TestCase
setUp and tearDown methods for test fixtures
Example test methods showing different types of assertions
The if __name__ == '__main__' block to run tests

To use this template:

Replace TestClassName with a meaningful name for your test class
Add your own test methods (they must start with test_)"""

    def tearDown(self):
        """
        This method is called after each test case.
        Use it to clean up any resources created in setUp.
        """

    def test_fit(self):
        """Test case demonstrating the ability to train the tokenizer"""
        tokenizer = BPETokenizer()
        tokenizer.fit(self.training_text)

    def test_encode(self):
        """The tokenizer should be able to compress the representation of a text"""
        tokenizer = BPETokenizer()
        tokenizer.fit(self.training_text)
        encoded = tokenizer.encode(self.test_text)
        assert len(encoded) < len(self.test_text)

    def test_decode(self):
        """The tokenizer should be such that decode(encode(text)) == text"""
        tokenizer = BPETokenizer()
        tokenizer.fit(self.training_text)
        decoded = tokenizer.decode(tokenizer.encode(self.test_text))
        assert decoded == self.test_text



if __name__ == '__main__':
    unittest.main()
