from lightning.pytorch.callbacks import Callback
import glob
import os
import torch

class PredictionOutputCallback(Callback):
    def __init__(self):
        self.outputs = []
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.outputs.append(outputs)
    
    def on_predict_epoch_end(self, trainer, pl_module):
        if not self.outputs:
            return
            
        # Get tokenizer from datamodule
        tokenizer = trainer.datamodule.tokenizer
        
        print("\nGenerated text:")
        print("="*50)
        # Process all outputs
        for output in self.outputs:
            decoded_text = tokenizer.decode(output[0].tolist())
            print("".join(decoded_text))
        print("="*50)
        
        # Clear outputs for next prediction
        self.outputs = []


class AutoLoadLastCheckpointCallback(Callback):
    def __init__(self, tokenizer_type: str = None):
        if tokenizer_type is not None:
            assert tokenizer_type in ["ASCII", "BPE"]
        self.tokenizer_type = tokenizer_type


    def on_predict_start(self, trainer, pl_module):
        if not trainer.ckpt_path:  # If no checkpoint was explicitly provided
            checkpoints_dir = "checkpoints"
            checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.ckpt"))
            if self.tokenizer_type is not None:
                checkpoint_files = [f for f in checkpoint_files if self.tokenizer_type in f]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                trainer.ckpt_path = latest_checkpoint                
                # FIXME: should use lightning api to load checkpoint instead of this hack
                checkpoint = torch.load(latest_checkpoint)
                pl_module.load_state_dict(checkpoint['state_dict'])
                print(f"Automatically loading checkpoint: {trainer.ckpt_path}")
            else:
                print("No checkpoint were found!!")