import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

import lightning_ir  # noqa: F401
from lightning_ir.lightning_utils.warmup_schedulers import (
    LR_SCHEDULERS,
    WarmupScheduler,
)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(Path.cwd()))


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


class CustomTrainer(Trainer):
    # TODO check that correct callbacks are registered for each subcommand
    def index(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        datamodule: LightningDataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
    ) -> List[Any] | List[List[Any]] | None:
        """Index a collection of documents."""
        return super().predict(
            model, dataloaders, datamodule, return_predictions, ckpt_path
        )

    def search(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        datamodule: LightningDataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
    ) -> List[Any] | List[List[Any]] | None:
        """Search for relevant documents."""
        return super().predict(
            model, dataloaders, datamodule, return_predictions, ckpt_path
        )

    def re_rank(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        datamodule: LightningDataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
    ) -> List[Any] | List[List[Any]] | None:
        """Re-rank a set of retrieved documents."""
        return super().predict(
            model, dataloaders, datamodule, return_predictions, ckpt_path
        )


class CustomLightningCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupScheduler | None = None,
    ) -> Any:
        if lr_scheduler is None:
            return optimizer

        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": lr_scheduler.interval}
        ]

    def add_arguments_to_parser(self, parser):
        parser.add_lr_scheduler_args(tuple(LR_SCHEDULERS))
        parser.link_arguments(
            "model.init_args.model_name_or_path", "data.init_args.model_name_or_path"
        )
        parser.link_arguments("model.init_args.config", "data.init_args.config")
        parser.link_arguments(
            "trainer.max_steps", "lr_scheduler.init_args.num_training_steps"
        )

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        return {
            **LightningCLI.subcommands(),
            "index": {"model", "dataloaders", "datamodule"},
            "search": {"model", "dataloaders", "datamodule"},
            "re_rank": {"model", "dataloaders", "datamodule"},
        }


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
        trainer_class=CustomTrainer,
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True},
    )


if __name__ == "__main__":
    main()
