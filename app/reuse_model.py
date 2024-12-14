"""Script to load and use trained model"""

import torch
from smolgpt.model.transformer import Transformer
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer
import json

def load_trained_model(checkpoint_path, config_path):
    # Load model configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model with saved configuration
    model = Transformer(
        config['vocab_size'],
        config['n_embed'],
        config['block_size'],
        config['n_heads'],
        config['n_layer'],
        config['dropout']
    )
    
    # Load state dict from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def generate_text(model, tokenizer, prompt="", max_tokens=1000, device="mps"):
    model = model.to(device)
    model.eval()
    
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    
    return tokenizer.decode(output)

if __name__ == "__main__":
    DEVICE = "mps"
    
    # Load tokenizer
    tokenizer = BPETokenizer.load("tokenizer/vocab/")
    
    # Load model
    model = load_trained_model(
        checkpoint_path="lightning_logs/version_9/checkpoints/epoch=0-step=5000.ckpt",  # Replace XXXX with actual step number
        config_path="checkpoints/model_config.json"
    )
    
    # Generate text
    output = generate_text(
        model,
        tokenizer,
        prompt="Once upon a time",  # Optional prompt
        max_tokens=1000,
        device=DEVICE
    )
    
    print("Generated text:")
    print(output)