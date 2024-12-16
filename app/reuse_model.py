from lightning.pytorch.cli import LightningCLI
import torch
from smolgpt.model.transformer import Transformer
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer
from smolgpt.tokenizer.ascii_tokenizer import ASCIITokenizer

class GenerationCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--checkpoint_path", type=str, required=True)
        parser.add_argument("--prompt", type=str, default="Once upon a time")
        parser.add_argument("--max_tokens", type=int, default=1000)
        parser.add_argument("--tokenizer_path", type=str, default="tokenizer/vocab/")

def generate_text(model, tokenizer, prompt="", max_tokens=1000):
    model.eval()
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=model.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=model.device)
    
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    return tokenizer.decode(output)

def main():
    # Initialize CLI
    cli = GenerationCLI(
        model_class=Transformer,
        save_config_callback=None,
        run=False
    )
    
    # Load tokenizer first to get vocab_size
    if cli.config["tokenizer_path"]:
        tokenizer = BPETokenizer.load(cli.config["tokenizer_path"])
    else:
        tokenizer = ASCIITokenizer()
    
    # Load checkpoint
    checkpoint = torch.load(
        cli.config["checkpoint_path"],
        map_location=torch.device('mps')
    )
    
    # Extract hyperparameters from checkpoint
    hparams = checkpoint['hyper_parameters']
    
    # Create model with hyperparameters from checkpoint
    model = Transformer(
        n_embed=hparams['n_embed'],
        block_size=hparams['block_size'],
        n_heads=hparams['n_heads'],
        n_layer=hparams['n_layer'],
        dropout=hparams['dropout'],
        learning_rate=hparams['learning_rate'],
        device_type=hparams['device_type'],
        vocab_size  =hparams['vocab_size']

    )
    
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set up model for generation
    model = model.to('mps')
    
    # Generate text
    output = generate_text(
        model,
        tokenizer,
        prompt=cli.config["prompt"],
        max_tokens=cli.config["max_tokens"]
    )
    
    print("\nGenerated text:")
    print(output)

if __name__ == "__main__":
    main()