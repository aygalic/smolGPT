from lightning.pytorch.callbacks import Callback

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
            print(decoded_text)
        print("="*50)
        
        # Clear outputs for next prediction
        self.outputs = []



