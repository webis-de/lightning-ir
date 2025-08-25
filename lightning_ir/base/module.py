"""LightningModule for Lightning IR.

This module contains the main module class deriving from a LightningModule_.

.. _LightningModule: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Self, Sequence, Tuple, Type

import pandas as pd
import torch
from lightning import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from transformers import BatchEncoding

from ..data import IRDataset, RankBatch, RunDataset, SearchBatch, TrainBatch
from ..loss import InBatchLossFunction, LossFunction
from .config import LightningIRConfig
from .model import LightningIRModel, LightningIROutput
from .tokenizer import LightningIRTokenizer
from .validation_utils import create_qrels_from_dicts, create_run_from_scores, evaluate_run


class LightningIRModule(LightningModule):
    """LightningIRModule base class. It dervies from a LightningModule_. LightningIRModules contain a
    LightningIRModel and a LightningIRTokenizer and implements the training, validation, and testing steps for the
    model. Derived classes must implement the forward method for the model.

    .. _LightningModule: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: LightningIRConfig | None = None,
        model: LightningIRModel | None = None,
        loss_functions: Sequence[LossFunction | Tuple[LossFunction, float]] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        """Initializes the LightningIRModule.

        .. _ir-measures: https://ir-measur.es/en/latest/index.html

        Args:
            model_name_or_path (str | None): Name or path of backbone model or fine-tuned Lightning IR model.
                Defaults to None.
            config (LightningIRConfig | None): LightningIRConfig to apply when loading from backbone model.
                Defaults to None.
            model (LightningIRModel | None): Already instantiated Lightning IR model. Defaults to None.
            loss_functions (Sequence[LossFunction | Tuple[LossFunction, float]] | None):
                Loss functions to apply during fine-tuning, optional loss weights can be provided per loss function
                Defaults to None.
            evaluation_metrics (Sequence[str] | None): Metrics corresponding to ir-measures_ measure strings
                to apply during validation or testing. Defaults to None.
            model_kwargs (Mapping[str, Any] | None): Additional keyword arguments to pass to `from_pretrained` when
                loading a model. Defaults to None.
        Raises:
            ValueError: If both model and model_name_or_path are provided.
            ValueError: If neither model nor model_name_or_path are provided.
        """
        super().__init__()
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.save_hyperparameters()
        if model is not None and model_name_or_path is not None:
            raise ValueError("Only one of model or model_name_or_path must be provided.")
        if model is None:
            if model_name_or_path is None:
                raise ValueError("Either model or model_name_or_path must be provided.")
            model = LightningIRModel.from_pretrained(model_name_or_path, config=config, **model_kwargs)

        self.model: LightningIRModel = model
        self.config = self.model.config
        self.loss_functions: List[Tuple[LossFunction, float]] | None = None
        if loss_functions is not None:
            self.loss_functions = []
            for loss_function in loss_functions:
                if isinstance(loss_function, LossFunction):
                    self.loss_functions.append((loss_function, 1.0))
                else:
                    self.loss_functions.append(loss_function)
        self.evaluation_metrics = evaluation_metrics
        self._optimizer: torch.optim.Optimizer | None = None
        self.tokenizer = LightningIRTokenizer.from_pretrained(self.config.name_or_path, config=self.config)
        self._additional_log_metrics: Dict[str, float] = {}

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""
        super().on_train_start()
        # NOTE huggingface models are in eval mode by default
        self.model = self.model.train()

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        # NOTE monkey patch result printing of the trainer
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is None:
            return

        trainer._evaluation_loop._print_results = lambda *args, **kwargs: None

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        self.on_validation_start()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optizmizer for fine-tuning. This method is ignored when using the CLI. When using Lightning IR
        programmatically, the optimizer must be set using :meth:`set_optimizer`.

        Returns:
            torch.optim.Optimizer: The optimizer set for the model.
        Raises:
            ValueError: If optimizer is not set. Call `set_optimizer`.
        """
        if self._optimizer is None:
            raise ValueError("Optimizer is not set. Call `set_optimizer`.")
        return self._optimizer

    def set_optimizer(self, optimizer: Type[torch.optim.Optimizer], **optimizer_kwargs: Dict[str, Any]) -> Self:
        """Sets the optimizer for the model. Necessary for fine-tuning when not using the CLI.

        Args:
            optimizer (Type[torch.optim.Optimizer]): Torch optimizer class.
            optimizer_kwargs (Dict[str, Any]): Arguments to initialize the optimizer.
        Returns:
            LightningIRModule: Self with the optimizer set.
        """
        self._optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        return self

    def score(self, queries: Sequence[str] | str, docs: Sequence[Sequence[str]] | Sequence[str]) -> LightningIROutput:
        """Computes relevance scores for queries and documents.

        Args:
            queries (Sequence[str] | str): Queries to score.
            docs (Sequence[Sequence[str]] | Sequence[str]): Documents to score.
        Returns:
            LightningIROutput: Model output containing the scores.
        """
        if isinstance(queries, str):
            queries = (queries,)
        if isinstance(docs[0], str):
            docs = (docs,)
        batch = RankBatch(queries, docs, None, None)
        with torch.no_grad():
            return self.forward(batch)

    def forward(self, batch: TrainBatch | RankBatch | SearchBatch) -> LightningIROutput:
        """Handles the forward pass of the model.

        Args:
            batch (TrainBatch | RankBatch | SearchBatch): Batch of training or ranking data.
        Returns:
            LightningIROutput: Model output.
        Raises:
            NotImplementedError: Must be implemented by derived class.
        """
        raise NotImplementedError

    def prepare_input(
        self, queries: Sequence[str] | None, docs: Sequence[str] | None, num_docs: Sequence[int] | int | None
    ) -> Dict[str, BatchEncoding]:
        """Tokenizes queries and documents and returns the tokenized BatchEncoding_.

        .. _BatchEncoding: https://huggingface.co/transformers/main_classes/tokenizer#transformers.BatchEncoding

        Args:
            queries (Sequence[str] | None): Queries to tokenize.
            docs (Sequence[str] | None): Documents to tokenize.
            num_docs (Sequence[int] | int | None): Number of documents per query, if None num_docs is inferred by
                `len(docs) // len(queries)`. Defaults to None.
        Returns:
            Dict[str, BatchEncoding]: Tokenized queries and documents, format depends on the tokenizer.
        """
        encodings = self.tokenizer.tokenize(
            queries, docs, return_tensors="pt", padding=True, truncation=True, num_docs=num_docs
        )
        for key in encodings:
            encodings[key] = encodings[key].to(self.device)
        return encodings

    def _compute_losses(self, batch: TrainBatch, output: LightningIROutput) -> List[torch.Tensor]:
        """Computes the losses for a training batch."""
        raise NotImplementedError

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        """Handles the training step for the model.

        Args:
            batch (TrainBatch): Batch of training data.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Sum of the losses weighted by the loss weights.
        Raises:
            ValueError: If no loss functions are set.
        """
        if self.loss_functions is None:
            raise ValueError("Loss functions are not set")
        output = self.forward(batch)
        losses = self._compute_losses(batch, output)
        total_loss = torch.tensor(0)
        assert len(losses) == len(self.loss_functions)
        for (loss_function, loss_weight), loss in zip(self.loss_functions, losses):
            self.log(loss_function.__class__.__name__, loss)
            total_loss = total_loss + loss * loss_weight
        self.log("loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(
        self, batch: TrainBatch | RankBatch | SearchBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> LightningIROutput:
        """Handles the validation step for the model.

        Args:
            batch (TrainBatch | RankBatch | SearchBatch): Batch of validation or testing data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        Returns:
            LightningIROutput: Model output.
        """
        output = self.forward(batch)

        if self.evaluation_metrics is None:
            return output

        dataset = self.get_dataset(dataloader_idx)
        dataset_id = str(dataloader_idx) if dataset is None else self.get_dataset_id(dataset)
        metrics = self.validate(output, batch)
        for key, value in metrics.items():
            key = f"{dataset_id}/{key}"
            self.log(key, value, batch_size=len(batch.queries))
        return output

    def test_step(
        self,
        batch: TrainBatch | RankBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> LightningIROutput:
        """Handles the testing step for the model. Passes the batch to the validation step.

        Args:
            batch (TrainBatch | RankBatch): Batch of testing data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        Returns:
            LightningIROutput: Model output.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def get_dataset(self, dataloader_idx: int) -> IRDataset | None:
        """Gets the dataset instance from the dataloader index. Returns None if no dataset is found.

        Args:
            dataloader_idx (int): Index of the dataloader.
        Returns:
            IRDataset | None: Inference dataset or None if no dataset is found.
        """
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is None:
            return None
        STAGE_TO_DATALOADER = {
            RunningStage.VALIDATING: "val_dataloaders",
            RunningStage.TESTING: "test_dataloaders",
            RunningStage.PREDICTING: "predict_dataloaders",
            RunningStage.SANITY_CHECKING: "val_dataloaders",
        }
        if trainer.state.stage is None:
            return None
        dataloaders = getattr(trainer, STAGE_TO_DATALOADER[trainer.state.stage], None)
        if dataloaders is None:
            return None
        if isinstance(dataloaders, torch.utils.data.DataLoader):
            dataloaders = [dataloaders]
        return dataloaders[dataloader_idx].dataset

    def get_dataset_id(self, dataset: IRDataset) -> str:
        """Gets the dataset id from the dataloader index for logging.

        .. _ir-datasets: https://ir-datasets.com/

        Args:
            dataset (IRDataset): Dataset instance.
        Returns:
            str: Path to run file, ir-datasets_ dataset id, or dataloader index.
        """
        if isinstance(dataset, RunDataset) and dataset.run_path is not None:
            dataset_id = dataset.run_path.name
        else:
            dataset_id = dataset.dataset_id
        return dataset_id

    def validate(
        self,
        output: LightningIROutput,
        batch: TrainBatch | RankBatch | SearchBatch,
    ) -> Dict[str, float]:
        """Validates the model output with the evaluation metrics and loss functions.

        Args:
            output (LightningIROutput): Model output.
            batch (TrainBatch | RankBatch | SearchBatch): Batch of validation or testing data.
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        metrics: Dict[str, float] = {}
        if self.evaluation_metrics is None or output.scores is None:
            return metrics
        metrics.update(self.validate_metrics(output, batch))
        metrics.update(self.validate_loss(output, batch))
        return metrics

    def validate_metrics(
        self,
        output: LightningIROutput,
        batch: TrainBatch | RankBatch | SearchBatch,
    ) -> Dict[str, float]:
        """Validates the model output with the evaluation metrics.

        Args:
            output (LightningIROutput): Model output.
            batch (TrainBatch | RankBatch | SearchBatch): Batch of validation or testing data.
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        Raises:
            ValueError: If query_ids or doc_ids are not set in the batch.
        """
        metrics: Dict[str, float] = {}
        qrels = batch.qrels
        if self.evaluation_metrics is None or qrels is None:
            return metrics
        query_ids = batch.query_ids
        doc_ids = batch.doc_ids
        if query_ids is None:
            raise ValueError("query_ids must be set")
        if doc_ids is None:
            raise ValueError("doc_ids must be set")
        evaluation_metrics = [metric for metric in self.evaluation_metrics if metric != "loss"]
        ir_measures_qrels = create_qrels_from_dicts(qrels)
        if evaluation_metrics and qrels is not None and output.scores is not None:
            run = create_run_from_scores(query_ids, doc_ids, output.scores)
            metrics.update(evaluate_run(run, ir_measures_qrels, evaluation_metrics))
        return metrics

    def validate_loss(
        self,
        output: LightningIROutput,
        batch: TrainBatch | RankBatch | SearchBatch,
    ) -> Dict[str, float]:
        """Validates the model output with the loss functions.

        Args:
            output (LightningIROutput): Model output.
            batch (TrainBatch | RankBatch | SearchBatch): Batch of validation or testing data.
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        metrics: Dict[str, float] = {}
        query_ids = batch.query_ids
        if query_ids is None:
            raise ValueError("query_ids must be set")
        if (
            self.evaluation_metrics is None
            or "loss" not in self.evaluation_metrics
            or getattr(batch, "targets", None) is None
            or self.loss_functions is None
            or output.scores is None
        ):
            return metrics
        output.scores = output.scores.view(len(query_ids), -1)
        for loss_function, _ in self.loss_functions:
            # NOTE skip in-batch losses because they can use a lot of memory
            if isinstance(loss_function, InBatchLossFunction):
                continue
            metrics[f"validation-{loss_function.__class__.__name__}"] = loss_function.compute_loss(output, batch)
        return metrics

    def on_validation_end(self) -> None:
        """Prints the validation results for each dataloader."""
        trainer = self.trainer
        if not (trainer.is_global_zero and trainer._evaluation_loop.verbose):
            return
        results = trainer.callback_metrics

        data = []
        for key, value in {**results, **self._additional_log_metrics}.items():
            if "dataloader_idx" in key:
                key = "/".join(key.split("/")[:-1])
            *dataset_parts, metric = key.split("/")
            if metric.startswith("validation-"):
                metric = metric[len("validation-") :]
            dataset = "/".join(dataset_parts)
            if isinstance(value, torch.Tensor):
                value = value.item()
            data.append({"dataset": dataset, "metric": metric, "value": value})
        if not data:
            return
        df = pd.DataFrame(data)
        df = df.pivot(index="dataset", columns="metric", values="value")
        df.columns.name = None

        # bring into correct order when skipping inference datasets
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "inference_datasets"):
            inference_datasets = datamodule.inference_datasets
            if len(inference_datasets) != df.shape[0]:
                raise ValueError(
                    "Number of inference datasets does not match number of dataloaders. "
                    "Check if the dataloaders are correctly configured."
                )
            dataset_ids = [self.get_dataset_id(dataset) for dataset in inference_datasets]
            df = df.reindex(dataset_ids)

        trainer.print(df)

    def on_test_end(self) -> None:
        """Prints the accumulated metrics for each dataloader."""
        self.on_validation_end()

    def save_pretrained(self, save_path: str | Path) -> None:
        """Saves the model and tokenizer to the save path.

        Args:
            save_path (str | Path): Path to save the model and tokenizer.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Saves the model and tokenizer to the trainer's log directory."""
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            _step = self.trainer.global_step
            self.config.save_step = _step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)
