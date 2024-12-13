import unittest
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer

class TestClassName(unittest.TestCase):
    def setUp(self):
        """
        This method is called before each test case.
        Use it to set up any necessary test fixtures.
        """
        # Initialize any objects or variables you'll need for testing
        self.training_text = """the full recipe that defines how your nn.Modules interact.

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
    
        self.test_text = """This template includes the essential components for writing unit tests in Python:

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
        # Clean up any resources
        pass
    
    def test_fit(self):
        """Test case demonstrating different assertion methods"""
        tokenizer = BPETokenizer()
        tokenizer.fit(self.training_text)

    def test_encode(self):
        """Test case demonstrating different assertion methods"""
        tokenizer = BPETokenizer()
        tokenizer.fit(self.training_text)
        encoded = tokenizer.encode(self.test_text)
        assert len(encoded) < len(self.test_text)

    def test_decode(self):
        """Test case demonstrating different assertion methods"""
        tokenizer = BPETokenizer()
        tokenizer.fit(self.training_text)
        decoded = tokenizer.decode(tokenizer.encode(self.test_text))
        assert decoded == self.test_text



if __name__ == '__main__':
    unittest.main()