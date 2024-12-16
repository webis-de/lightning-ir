import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

import lightning_ir  # noqa: F401
from lightning_ir.schedulers.lr_schedulers import LR_SCHEDULERS, WarmupLRScheduler

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

sys.path.append(str(Path.cwd()))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LightningIRSaveConfigCallback(SaveConfigCallback):
    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit" or trainer.logger is None:
            return
        return super().setup(trainer, pl_module, stage)


class LightningIRWandbLogger(WandbLogger):
    @property
    def save_dir(self) -> str | None:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        if isinstance(self.experiment, DummyExperiment):
            return None
        return self.experiment.dir


class LightningIRTrainer(Trainer):
    # TODO check that correct callbacks are registered for each subcommand

    def index(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> List[Mapping[str, float]]:
        """Index a collection of documents."""
        return super().test(model, dataloaders, ckpt_path, verbose, datamodule)

    def search(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> List[Mapping[str, float]]:
        """Search for relevant documents."""
        return super().test(model, dataloaders, ckpt_path, verbose, datamodule)

    def re_rank(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> List[Mapping[str, float]]:
        """Re-rank a set of retrieved documents."""
        return super().test(model, dataloaders, ckpt_path, verbose, datamodule)


class LightningIRCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupLRScheduler | None = None,
    ) -> Any:
        if lr_scheduler is None:
            return optimizer

        return [optimizer], [{"scheduler": lr_scheduler, "interval": lr_scheduler.interval}]

    def add_arguments_to_parser(self, parser):
        parser.add_lr_scheduler_args(tuple(LR_SCHEDULERS))
        parser.link_arguments("model.init_args.model_name_or_path", "data.init_args.model_name_or_path")
        parser.link_arguments("model.init_args.config", "data.init_args.config")
        parser.link_arguments("trainer.max_steps", "lr_scheduler.init_args.num_training_steps")

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        return {
            "fit": LightningCLI.subcommands()["fit"],
            "index": {"model", "dataloaders", "datamodule"},
            "search": {"model", "dataloaders", "datamodule"},
            "re_rank": {"model", "dataloaders", "datamodule"},
        }

    def _add_configure_optimizers_method_to_model(self, subcommand: str | None) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super()._add_configure_optimizers_method_to_model(subcommand)


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
                --trainer.logger LightningIRWandbLogger \
                --print_config > default.yaml

        To run with the default config:

            python main.py fit \
                --config default.yaml

    """
    LightningIRCLI(
        trainer_class=LightningIRTrainer,
        save_config_callback=LightningIRSaveConfigCallback,
        save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True},
    )


if __name__ == "__main__":
    main()
