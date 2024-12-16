from lightning.pytorch.cli import LightningCLI
from smolgpt.data.data_module import TinyShakespeareData
from smolgpt.model.transformer import Transformer
from smolgpt.tokenizer.bpe_tokenizer import BPETokenizer
from smolgpt.tokenizer.ascii_tokenizer import ASCIITokenizer

def cli_main():
    # The LightningCLI will automatically create the Trainer from config
    cli = LightningCLI(
        model_class=Transformer,
        datamodule_class=TinyShakespeareData,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "yaml"}
    )
    if cli.subcommand == "predict":
        output = cli.trainer.predict(cli.model, cli.datamodule)[0][0].tolist()
        print("\nAll generated texts:")
        print("="*50)
        if "tokenizer_path" in cli.config.keys():
            tokenizer = BPETokenizer.load(cli.config["tokenizer_path"])
        else:
            tokenizer = ASCIITokenizer()

        print("".join(tokenizer.decode(output)))

if __name__ == "__main__":
    cli_main()