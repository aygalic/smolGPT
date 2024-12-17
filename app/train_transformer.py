from lightning.pytorch.cli import LightningCLI
from smolgpt.data.data_module import TinyShakespeareData
from smolgpt.model.transformer import Transformer
from smolgpt.utils.custom_callback import PredictionOutputCallback

def cli_main():
    # Create the prediction callback
    prediction_callback = PredictionOutputCallback()
    
    # Add it to the CLI
    cli = LightningCLI(
        model_class=Transformer,
        datamodule_class=TinyShakespeareData,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "yaml"}
    )
    
    if cli.subcommand == "predict":
        # Add the callback for prediction
        cli.trainer.callbacks.append(prediction_callback)
        # Run prediction once
        cli.trainer.predict(cli.model, cli.datamodule)

if __name__ == "__main__":
    cli_main()