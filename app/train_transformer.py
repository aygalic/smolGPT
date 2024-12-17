from lightning.pytorch.cli import LightningCLI
from smolgpt.data.data_module import TinyShakespeareData
from smolgpt.model.transformer import Transformer

def cli_main():
    cli = LightningCLI(
        model_class=Transformer,
        datamodule_class=TinyShakespeareData,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "yaml"}
    )

if __name__ == "__main__":
    cli_main()