"""Main entry point for Lightning IR using the Lightning IR CLI.

The module also defines several helper classes for configuring and running experiments.
"""

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from textwrap import dedent
from typing import Any

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
    """Lightning IR configuration saving callback with intelligent save conditions.

    This callback extends PyTorch Lightning's SaveConfigCallback_ to provide smarter configuration
    file saving behavior specifically designed for Lightning IR workflows. It only saves YAML
    configuration files during the 'fit' stage and when a logger is properly configured, preventing
    unnecessary file creation during inference operations like indexing, searching, or re-ranking.

    The callback automatically saves the complete experiment configuration including model, data,
    trainer, and optimizer settings to enable full experiment reproducibility.

    .. _SaveConfigCallback: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.SaveConfigCallback.html

    Examples:
        Automatic usage through LightningIRCLI:

        .. code-block:: python

            from lightning_ir.main import LightningIRCLI, LightningIRSaveConfigCallback

            # The callback is automatically configured in the CLI
            cli = LightningIRCLI(
                save_config_callback=LightningIRSaveConfigCallback,
                save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True}
            )

        Manual usage with trainer:

        .. code-block:: python

            from lightning_ir import LightningIRTrainer, LightningIRSaveConfigCallback

            # Add callback to trainer
            callback = LightningIRSaveConfigCallback(
                config_filename="experiment_config.yaml",
                overwrite=True
            )
            trainer = LightningIRTrainer(callbacks=[callback])

        Configuration file output example:

        .. code-block:: yaml

            # Generated pl_config.yaml
            model:
              class_path: lightning_ir.BiEncoderModule
              init_args:
                model_name_or_path: bert-base-uncased
            data:
              class_path: lightning_ir.LightningIRDataModule
              init_args:
                train_batch_size: 32
            trainer:
              max_steps: 100000
    """

    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup the callback with intelligent save conditions.

        This method implements the core logic for conditional configuration saving. It only
        proceeds with configuration file saving when both conditions are met:
        1. The training stage is 'fit' (not inference stages like index/search/re_rank)
        2. A logger is properly configured on the trainer

        This prevents unnecessary configuration file creation during inference operations
        while ensuring that training experiments are properly documented for reproducibility.

        Args:
            trainer (Trainer): The Lightning trainer instance containing training configuration
                and logger settings.
            pl_module (LightningModule): The Lightning module instance being trained or used
                for inference.
            stage (str): The current training stage. Expected values include 'fit', 'validate',
                'test', 'predict', as well as Lightning IR specific stages like 'index',
                'search', 're_rank'.

        Examples:
            The method automatically handles different stages:

            .. code-block:: python

                # During training - config will be saved
                trainer.fit(module, datamodule)  # stage='fit', saves config

                # During inference - config will NOT be saved
                trainer.index(module, datamodule)    # stage='index', skips saving
                trainer.search(module, datamodule)   # stage='search', skips saving
                trainer.re_rank(module, datamodule)  # stage='re_rank', skips saving
        """
        if stage != "fit" or trainer.logger is None:
            return
        super().setup(trainer, pl_module, stage)


class LightningIRWandbLogger(WandbLogger):
    """Lightning IR extension of the Weights & Biases Logger for enhanced experiment tracking.

    This logger extends the PyTorch Lightning WandbLogger_ to provide improved file management
    and experiment tracking specifically tailored for Lightning IR experiments. It ensures that
    experiment files are properly saved in the WandB run's files directory and handles the
    save directory management correctly.

    .. _WandbLogger: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.WandbLogger.html
    """

    @property
    def save_dir(self) -> str | None:
        """Gets the save directory for experiment files and artifacts.

        This property returns the directory where WandB saves experiment files, logs, and
        artifacts. It handles the case where the experiment might not be properly initialized
        (DummyExperiment) and returns None in such cases to prevent errors.

        Returns:
            str | None: The absolute path to the WandB experiment directory where files
                       are saved, or None if the experiment is not properly initialized
                       or WandB is running in offline/disabled mode.
        """
        if isinstance(self.experiment, DummyExperiment):
            return None
        return self.experiment.dir


class LightningIRTrainer(Trainer):
    """Lightning IR Trainer that extends PyTorch Lightning Trainer with information retrieval specific methods.

    This trainer inherits all functionality from the PyTorch Lightning Trainer_ and adds specialized methods
    for information retrieval tasks including document indexing, searching, and re-ranking. It provides a
    unified interface for both training neural ranking models and performing inference across different
    IR stages.

    The trainer seamlessly integrates with Lightning IR callbacks and supports all standard Lightning features
    including distributed training, mixed precision, gradient accumulation, and checkpointing.

    .. _PyTorch Lightning Trainer: https://lightning.ai/docs/pytorch/stable/common/trainer.html

    Examples:
        Basic usage for fine-tuning and inference:

        .. code-block:: python

            from lightning_ir import LightningIRTrainer, BiEncoderModule, LightningIRDataModule

            # Initialize trainer with Lightning configuration
            trainer = LightningIRTrainer(
                max_steps=100_000,
                precision="16-mixed",
                devices=2,
                accelerator="gpu"
            )

            # Fine-tune a model
            module = BiEncoderModule(model_name_or_path="bert-base-uncased")
            datamodule = LightningIRDataModule(...)
            trainer.fit(module, datamodule)

            # Index documents
            trainer.index(module, datamodule)

            # Search for relevant documents
            trainer.search(module, datamodule)

            # Re-rank retrieved documents
            trainer.re_rank(module, datamodule)

    Note:
        The trainer requires appropriate callbacks to be configured for each IR task:
        - IndexCallback for indexing operations
        - SearchCallback for search operations
        - ReRankCallback for re-ranking operations
    """

    # TODO check that correct callbacks are registered for each subcommand

    def index(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> list[Mapping[str, float]]:
        """Index a collection of documents using a fine-tuned bi-encoder model.

        This method performs document indexing by running inference on a document collection and
        storing the resulting embeddings in an index structure. It requires an IndexCallback to
        be configured in the trainer to handle the actual indexing process.

        Args:
            model (LightningModule | None): The LightningIRModule containing the bi-encoder model
                to use for encoding documents. If None, uses the model from the datamodule.
            dataloaders (Any | LightningDataModule | None): DataLoader(s) or LightningIRDataModule
                containing the document collection to index. Should contain DocDataset instances.
            ckpt_path (str | Path | None): Path to a model checkpoint to load before indexing.
                If None, uses the current model state.
            verbose (bool): Whether to display progress during indexing. Defaults to True.
            datamodule (LightningDataModule | None): LightningIRDataModule instance. Alternative
                to passing dataloaders directly.

        Returns:
            list[Mapping[str, float]]: list of dictionaries containing indexing metrics and results.

        Example:
            .. code-block:: python

                from lightning_ir import LightningIRTrainer, BiEncoderModule, LightningIRDataModule
                from lightning_ir import IndexCallback, TorchDenseIndexConfig, DocDataset

                # Setup trainer with index callback
                callback = IndexCallback(
                    index_dir="./index",
                    index_config=TorchDenseIndexConfig()
                )
                trainer = LightningIRTrainer(callbacks=[callback])

                # Setup model and data
                module = BiEncoderModule(model_name_or_path="webis/bert-bi-encoder")
                datamodule = LightningIRDataModule(
                    inference_datasets=[DocDataset("msmarco-passage")]
                )

                # Index the documents
                trainer.index(module, datamodule)

        Note:
            - Requires IndexCallback to be configured in trainer callbacks
            - Only works with bi-encoder models that can encode documents
            - The index type and configuration are specified in the IndexCallback
        """
        return super().test(model, dataloaders, ckpt_path, verbose, datamodule)

    def search(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> list[Mapping[str, float]]:
        """Search for relevant documents using a bi-encoder model and pre-built index.

        This method performs dense or sparse retrieval by encoding queries and searching through
        a pre-built index to find the most relevant documents. It requires a SearchCallback to
        be configured in the trainer to handle the search process and optionally a RankCallback
        to save results.

        Args:
            model (LightningModule | None): The LightningIRModule containing the bi-encoder model
                to use for encoding queries. If None, uses the model from the datamodule.
            dataloaders (Any | LightningDataModule | None): DataLoader(s) or LightningIRDataModule
                containing the queries to search for. Should contain QueryDataset instances.
            ckpt_path (str | Path | None): Path to a model checkpoint to load before searching.
                If None, uses the current model state.
            verbose (bool): Whether to display progress during searching. Defaults to True.
            datamodule (LightningDataModule | None): LightningIRDataModule instance. Alternative
                to passing dataloaders directly.

        Returns:
            list[Mapping[str, float]]: list of dictionaries containing search metrics and effectiveness
                results (if relevance judgments are available).

        Example:
            .. code-block:: python

                from lightning_ir import LightningIRTrainer, BiEncoderModule, LightningIRDataModule
                from lightning_ir import SearchCallback, RankCallback, QueryDataset
                from lightning_ir import TorchDenseSearchConfig

                # Setup trainer with search and rank callbacks
                search_callback = SearchCallback(
                    index_dir="./index",
                    search_config=TorchDenseSearchConfig(k=100)
                )
                rank_callback = RankCallback(results_dir="./results")
                trainer = LightningIRTrainer(callbacks=[search_callback, rank_callback])

                # Setup model and data
                module = BiEncoderModule(model_name_or_path="webis/bert-bi-encoder")
                datamodule = LightningIRDataModule(
                    inference_datasets=[QueryDataset("trec-dl-2019/queries")]
                )

                # Search for relevant documents
                results = trainer.search(module, datamodule)

        Note:
            - Requires SearchCallback to be configured in trainer callbacks
            - Index must be built beforehand using the index() method
            - Search configuration must match the index configuration used during indexing
            - Add RankCallback to save search results to disk
        """
        return super().test(model, dataloaders, ckpt_path, verbose, datamodule)

    def re_rank(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> list[Mapping[str, float]]:
        """Re-rank a set of retrieved documents using bi-encoder or cross-encoder models.

        This method performs re-ranking by scoring query-document pairs and reordering them
        based on relevance scores. Cross-encoders typically provide higher effectiveness for
        re-ranking tasks compared to bi-encoders. It requires a ReRankCallback to be configured
        in the trainer to handle saving the re-ranked results.

        Args:
            model (LightningModule | None): The LightningIRModule containing the model to use for
                re-ranking. Can be either BiEncoderModule or CrossEncoderModule. If None, uses
                the model from the datamodule.
            dataloaders (Any | LightningDataModule | None): DataLoader(s) or LightningIRDataModule
                containing the query-document pairs to re-rank. Should contain RunDataset instances.
            ckpt_path (str | Path | None): Path to a model checkpoint to load before re-ranking.
                If None, uses the current model state.
            verbose (bool): Whether to display progress during re-ranking. Defaults to True.
            datamodule (LightningDataModule | None): LightningIRDataModule instance. Alternative
                to passing dataloaders directly.

        Returns:
            list[Mapping[str, float]]: list of dictionaries containing re-ranking metrics and
                effectiveness results (if relevance judgments are available).

        Example:
            .. code-block:: python

                from lightning_ir import LightningIRTrainer, CrossEncoderModule, LightningIRDataModule
                from lightning_ir import ReRankCallback, RunDataset

                # Setup trainer with re-rank callback
                rerank_callback = ReRankCallback(results_dir="./reranked_results")
                trainer = LightningIRTrainer(callbacks=[rerank_callback])

                # Setup model and data
                module = CrossEncoderModule(model_name_or_path="webis/bert-cross-encoder")
                datamodule = LightningIRDataModule(
                    inference_datasets=[RunDataset("path/to/run/file.txt")]
                )

                # Re-rank the documents
                results = trainer.re_rank(module, datamodule)

        Note:
            - Requires ReRankCallback to be configured in trainer callbacks
            - Input data should be in run file format (query-document pairs with initial scores)
            - Cross-encoders typically provide better effectiveness than bi-encoders for re-ranking
        """
        return super().test(model, dataloaders, ckpt_path, verbose, datamodule)


class LightningIRCLI(LightningCLI):
    """Lightning IR Command Line Interface that extends PyTorch LightningCLI_ for information retrieval tasks.

    This CLI provides a unified command-line interface for fine-tuning neural ranking models and running
    information retrieval experiments. It extends the PyTorch LightningCLI_ with IR-specific subcommands
    and automatic configuration management for seamless integration between models, data, and training.

    .. _LightningCLI: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html

    Examples:
        Command line usage:

        .. code-block:: bash

            # Fine-tune a model
            lightning-ir fit --config fine-tune.yaml

            # Index documents
            lightning-ir index --config index.yaml

            # Search for documents
            lightning-ir search --config search.yaml

            # Re-rank documents
            lightning-ir re_rank --config re-rank.yaml

            # Generate default configuration
            lightning-ir fit --print_config > config.yaml

        Programmatic usage:

        .. code-block:: python

            from lightning_ir.main import LightningIRCLI, LightningIRTrainer, LightningIRSaveConfigCallback

            # Create CLI instance
            cli = LightningIRCLI(
                trainer_class=LightningIRTrainer,
                save_config_callback=LightningIRSaveConfigCallback,
                save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True}
            )

        YAML configuration example:

        .. code-block:: yaml

            model:
              class_path: lightning_ir.BiEncoderModule
              init_args:
                model_name_or_path: bert-base-uncased
                loss_functions:
                  - class_path: lightning_ir.InBatchCrossEntropy

            data:
              class_path: lightning_ir.LightningIRDataModule
              init_args:
                train_dataset:
                  class_path: lightning_ir.TupleDataset
                  init_args:
                    dataset_id: msmarco-passage/train/triples-small
                train_batch_size: 32

            trainer:
              max_steps: 100000
              precision: "16-mixed"

            optimizer:
              class_path: torch.optim.AdamW
              init_args:
                lr: 5e-5

    Note:
        - Automatically links model and data configurations (model_name_or_path, config)
        - Links trainer max_steps to learning rate scheduler num_training_steps
        - Supports all PyTorch Lightning CLI features including class path instantiation
        - Built-in support for warmup learning rate schedulers
        - Saves configuration files automatically during training
    """

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupLRScheduler | None = None,
    ) -> Any:
        """Configure optimizers and learning rate schedulers for Lightning training.

        This method automatically configures the optimizer and learning rate scheduler combination
        for Lightning training. It handles warmup learning rate schedulers by setting the
        appropriate interval and returning the correct format expected by Lightning.

        Args:
            lightning_module (LightningModule): The Lightning module being trained.
            optimizer (torch.optim.Optimizer): The optimizer instance to use for training.
            lr_scheduler (WarmupLRScheduler | None): Optional warmup learning rate scheduler.
                If None, only the optimizer is returned.

        Returns:
            Any: Either the optimizer alone (if no scheduler) or a tuple of optimizers and
                 schedulers list in Lightning's expected format.

        Note:
            - Warmup schedulers automatically set the correct interval based on scheduler type
            - Returns format compatible with Lightning's configure_optimizers method
        """
        if lr_scheduler is None:
            return optimizer

        return [optimizer], [{"scheduler": lr_scheduler, "interval": lr_scheduler.interval}]

    def add_arguments_to_parser(self, parser):
        """Add Lightning IR specific arguments and links to the CLI parser.

        This method extends the base Lightning CLI parser with IR-specific learning rate
        schedulers and automatically links related configuration arguments to ensure
        consistency between model, data, and trainer configurations.

        Args:
            parser: The CLI argument parser to extend.

        Note:
            Automatic argument linking:
            - model.init_args.model_name_or_path -> data.init_args.model_name_or_path
            - model.init_args.config -> data.init_args.config
            - trainer.max_steps -> lr_scheduler.init_args.num_training_steps
        """
        parser.add_lr_scheduler_args(tuple(LR_SCHEDULERS))
        parser.link_arguments("model.init_args.model_name_or_path", "data.init_args.model_name_or_path")
        parser.link_arguments("model.init_args.config", "data.init_args.config")
        parser.link_arguments("trainer.max_steps", "lr_scheduler.init_args.num_training_steps")

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Defines the list of available subcommands and the arguments to skip.

        Returns a dictionary mapping subcommand names to the set of configuration sections
        they require. This extends the base Lightning CLI with IR-specific subcommands for
        indexing, searching, and re-ranking operations.

        Returns:
            dict[str, set[str]]: Dictionary mapping subcommand names to required config sections.
                - fit: Standard Lightning training subcommand with all sections
                - index: Document indexing requiring model, dataloaders, and datamodule
                - search: Document search requiring model, dataloaders, and datamodule
                - re_rank: Document re-ranking requiring model, dataloaders, and datamodule

        """
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
    """Entry point for the Lightning IR command line interface.

    Initializes and runs the LightningIRCLI with the LightningIRTrainer and configuration
    callback. This function serves as the main entry point when Lightning IR is run from
    the command line using the 'lightning-ir' command.

    The CLI is configured with:
    - LightningIRTrainer as the trainer class for all IR operations
    - LightningIRSaveConfigCallback for automatic config file saving during training
    - Configuration to save configs as 'pl_config.yaml' with overwrite enabled

    Examples:
        This function is called when using Lightning IR from command line:

        .. code-block:: bash

            lightning-ir fit --config fine-tune.yaml
            lightning-ir index --config index.yaml
            lightning-ir search --config search.yaml
            lightning-ir re_rank --config re-rank.yaml

    Note:
        - Configuration files are automatically saved during fit operations
        - All PyTorch Lightning CLI features are available
        - Supports YAML configuration files and command line argument overrides
    """
    help_epilog = dedent(
        """\
        For full documentation, visit: https://lightning-ir.webis.de
        """
    )

    LightningIRCLI(
        trainer_class=LightningIRTrainer,
        save_config_callback=LightningIRSaveConfigCallback,
        save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True},
        parser_kwargs={
            "epilog": help_epilog,
            "formatter_class": argparse.RawDescriptionHelpFormatter,
        },
    )


if __name__ == "__main__":
    main()
