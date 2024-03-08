import os

import torch
from lightning import LightningModule, Trainer
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

import mvr.callbacks  # noqa
import mvr.colbert  # noqa
import mvr.datamodule  # noqa
import mvr.module  # noqa
import mvr.tide  # noqa
import mvr.xtr  # noqa

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomSaveConfigCallback(SaveConfigCallback):
    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit" or trainer.logger is None:
            return
        return super().setup(trainer, pl_module, stage)


class CustomWandbLogger(WandbLogger):
    @property
    def save_dir(self) -> str | None:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        if isinstance(self.experiment, DummyExperiment):
            return None
        return self.experiment.dir


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "model.init_args.model_name_or_path", "data.init_args.model_name_or_path"
        )
        parser.link_arguments("model.init_args.config", "data.init_args.config")


def main():
    """
    generate config using `python main.py fit --print_config > config.yaml`
    additional callbacks at:
    https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks

    Example:
        To obtain a default config:

            python main.py fit \
                --trainer.callbacks=ModelCheckpoint \
                --optimizer AdamW \
                --trainer.logger CustomWandbLogger \
                --print_config > default.yaml

        To run with the default config:

            python main.py fit \
                --config default.yaml

    """
    CustomLightningCLI(
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True},
    )


if __name__ == "__main__":
    main()
