from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Sequence

import torch
from lightning import LightningModule
from transformers import AutoConfig, AutoModel, BatchEncoding

from ..loss.loss import InBatchLossFunction, LossFunction
from . import (
    LightningIRConfig,
    LightningIRModel,
    LightningIRModelClassFactory,
    LightningIROutput,
)
from .validation_utils import (
    create_qrels_from_dicts,
    create_run_from_scores,
    evaluate_run,
)

if TYPE_CHECKING:
    from ..data import RankBatch, TrainBatch


class LightningIRModule(LightningModule):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: LightningIRConfig | None = None,
        model: LightningIRModel | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__()
        if model is not None and model_name_or_path is not None:
            raise ValueError(
                "Only one of model or model_name_or_path must be provided."
            )
        if model is None:
            if model_name_or_path is None:
                raise ValueError("Either model or model_name_or_path must be provided.")
            TransformerModel = AutoModel._model_mapping[
                AutoConfig.from_pretrained(model_name_or_path).__class__
            ]
            DerivedLightningIRModel = LightningIRModelClassFactory(
                TransformerModel, None if config is None else config.__class__
            )
            ir_config = None
            if config is not None:
                ir_config = DerivedLightningIRModel.config_class.from_pretrained(
                    model_name_or_path
                )
                ir_config.update(config.to_added_args_dict())
            model = DerivedLightningIRModel.from_pretrained(
                model_name_or_path, config=ir_config
            )

        self.model: LightningIRModel = model
        self.config = self.model.config
        self.loss_functions = loss_functions
        self.evaluation_metrics = evaluation_metrics
        self.tokenizer = self.config.__class__.tokenizer_class.from_pretrained(
            self.config.name_or_path, **self.config.to_tokenizer_dict()
        )

    def on_fit_start(self) -> None:
        self.train()
        return super().on_fit_start()

    def forward(self, batch: TrainBatch | RankBatch) -> LightningIROutput:
        raise NotImplementedError

    def prepare_input(
        self,
        queries: Sequence[str] | None,
        docs: Sequence[str] | None,
        num_docs: Sequence[int] | int,
    ) -> Dict[str, BatchEncoding]:
        encodings = self.tokenizer.tokenize(
            queries,
            docs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            num_docs=num_docs,
        )
        for key in encodings:
            encodings[key] = encodings[key].to(self.device)
        return encodings

    def compute_losses(
        self,
        batch: TrainBatch,
        loss_functions: Sequence[LossFunction] | None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        if self.loss_functions is None:
            raise ValueError("Loss function is not set")
        losses = self.compute_losses(batch, self.loss_functions)
        for key, loss in losses.items():
            self.log(key, loss)
        loss = sum(losses.values(), torch.tensor(0))
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: TrainBatch | RankBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> LightningIROutput:
        output = self.forward(batch)

        if self.evaluation_metrics is None:
            return output

        dataset_id = self.get_dataset_id(dataloader_idx)
        metrics = self.compute_metrics(batch, output)
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
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def get_dataset_id(self, dataloader_idx: int) -> str:
        dataset_id = str(dataloader_idx)
        datamodule = None
        try:
            datamodule = getattr(self.trainer, "datamodule", None)
            dataset_id = datamodule.inference_datasets[dataloader_idx].dataset_id
        except Exception:
            pass
        return dataset_id

    def compute_metrics(
        self, batch: TrainBatch | RankBatch, ouput: LightningIROutput
    ) -> Dict[str, float]:
        metrics = {}
        if self.evaluation_metrics is None:
            return metrics
        scores = ouput.scores
        query_ids = batch.query_ids
        doc_ids = batch.doc_ids
        if scores is None:
            raise ValueError("Scores are not provided")
        if query_ids is None:
            query_ids = tuple(str(i) for i in range(len(batch.queries)))
        if doc_ids is None:
            doc_ids = tuple(
                tuple(str(i + j) for j in range(len(docs)))
                for i, docs in enumerate(batch.docs)
            )

        targets = getattr(batch, "targets", None)
        if "loss" in self.evaluation_metrics and targets is not None:
            scores = scores.view(len(query_ids), -1)
            metrics.update(self.validate_loss(scores, targets))

        evaluation_metrics = [
            metric for metric in self.evaluation_metrics if metric != "loss"
        ]
        qrels = None if batch.qrels is None else create_qrels_from_dicts(batch.qrels)
        if evaluation_metrics and qrels is not None:
            run = create_run_from_scores(query_ids, doc_ids, scores)
            metrics.update(evaluate_run(run, qrels, evaluation_metrics))
        return metrics

    def validate_loss(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        metrics = {}
        if self.loss_functions is None:
            raise ValueError("Loss function is not set")
        for loss_function in self.loss_functions:
            # NOTE skip in-batch losses because they can use a lot of memory
            if isinstance(loss_function, InBatchLossFunction):
                continue
            metrics[f"validation-{loss_function.__class__.__name__}"] = (
                loss_function.compute_loss(scores, targets).item()
            )
        return metrics

    def on_validation_epoch_end(self) -> None:
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None:
            metrics = trainer.callback_metrics
            accum_metrics = defaultdict(list)
            for key, value in metrics.items():
                accum_metrics["/".join(key.split("/")[1:-1])].append(value)
            for key, value in accum_metrics.items():
                self.log(key, torch.stack(value).mean(), logger=False)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def save_pretrained(self, save_path: str | Path) -> None:
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            _step = self.trainer.global_step
            self.config.save_step = _step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)
