import click
import lightning as L
from pathlib import Path
from loguru import logger

from .pepinter import ESMSeqTokenizer
from .lightning import (
    PepInterForClassification,
    PepInterForAffinity,
    PepInterInferDataModuleForClassification,
    PepInterInferDataModuleForAffinity,
    CSVWriterCallback,
    CSVWriterCallbackForAffinity,
)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input_path", type=str, required=True)
@click.option("--output_path", type=str, required=True)
@click.option("--model_path", type=str, required=True)
@click.option("--task", type=str, required=True)
@click.option("--seed", type=int, default=42)
@click.option("--gpus", type=int, default=1)
def predict(input_path, output_path, model_path, task, seed, gpus):
    L.seed_everything(seed)
    input_path = Path(input_path)
    assert input_path.exists(), f"Input path {input_path} does not exist"

    output_path = Path(output_path)
    assert output_path.parent.exists(), f"Output path {output_path} does not exist"

    model_path = Path(model_path)
    assert model_path.exists(), f"Model path {model_path} does not exist"

    tokenizer = ESMSeqTokenizer()
    if task == "cls":
        pepinter = PepInterForClassification.load_from_checkpoint(
            checkpoint_path=model_path,
            tokenizer=tokenizer,
            strict=False,
            weights_only=False,
            map_location="cpu",
        )
        dm = PepInterInferDataModuleForClassification(
            data_path=input_path,
            tokenizer=tokenizer,
            batch_size=128,
            num_workers=8,
            max_len=1024,
        )
        csv_callback = CSVWriterCallback(
            save_dir=output_path.parent, filename="cls_pred.csv"
        )
    elif task == "affinity":
        pepinter = PepInterForAffinity.load_from_checkpoint(
            checkpoint_path=model_path,
            tokenizer=tokenizer,
            strict=False,
            weights_only=False,
            map_location="cpu",
        )
        dm = PepInterInferDataModuleForAffinity(
            data_path=input_path,
            tokenizer=tokenizer,
            batch_size=128,
            num_workers=8,
            max_len=1024,
        )
        csv_callback = CSVWriterCallbackForAffinity(
            save_dir=output_path.parent, filename="affinity_pred.csv"
        )
    else:
        raise ValueError(f"Invalid task: {task}")

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=gpus,
        callbacks=[
            L.pytorch.callbacks.RichProgressBar(),
            csv_callback,
        ],
    )
    trainer.predict(pepinter, dm)


if __name__ == "__main__":
    cli()
