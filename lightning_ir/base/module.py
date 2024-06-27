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
    from ..data import TrainBatch


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

        self.validation_step_outputs = defaultdict(lambda: defaultdict(list))

    def on_fit_start(self) -> None:
        self.train()
        return super().on_fit_start()

    def forward(self, batch: TrainBatch) -> LightningIROutput:
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
        batch: TrainBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.evaluation_metrics is None:
            return
        with torch.inference_mode():
            output = self.forward(batch)

        dataset_id = str(dataloader_idx)
        trainer = None
        try:
            trainer = self.trainer
        except RuntimeError:
            pass

        if trainer is not None and trainer.val_dataloaders is not None:
            dataset = trainer.val_dataloaders[dataloader_idx].dataset
            dataset_id = dataset.dataset_id

        self.validation_step_outputs[dataset_id]["scores"].append(output.scores)
        self.validation_step_outputs[dataset_id]["query_ids"].append(batch.query_ids)
        self.validation_step_outputs[dataset_id]["doc_ids"].append(batch.doc_ids)
        self.validation_step_outputs[dataset_id]["qrels"].append(batch.qrels)
        if hasattr(batch, "targets"):
            self.validation_step_outputs[dataset_id]["targets"].append(batch.targets)

    def on_validation_epoch_end(self) -> Dict[str, float] | None:
        if self.evaluation_metrics is None:
            return
        metrics = {}
        average_metrics = defaultdict(list)
        for dataset_id in self.validation_step_outputs:
            outputs = self.validation_step_outputs[dataset_id]
            query_ids = sum(outputs["query_ids"], [])
            doc_ids = sum(outputs["doc_ids"], [])
            scores = torch.cat(outputs["scores"])
            run = create_run_from_scores(query_ids, doc_ids, scores)

            if "loss" in self.evaluation_metrics:
                scores = scores.view(len(query_ids), -1)
                if "targets" not in outputs:
                    raise ValueError(
                        "Targets are not provided for validation loss calculation."
                    )
                targets = torch.cat(outputs["targets"]).view(*scores.shape, -1)
                metrics.update(self.validate_loss(dataset_id, scores, targets))

            evaluation_metrics = [
                metric for metric in self.evaluation_metrics if metric != "loss"
            ]
            if evaluation_metrics:
                qrels = create_qrels_from_dicts(sum(outputs["qrels"], []))
                for metric, value in evaluate_run(
                    run, qrels, evaluation_metrics
                ).items():
                    metrics[f"{dataset_id}/{metric}"] = value
                    average_metrics[metric].append(value)

        for key, value in metrics.items():
            self.log(key, value)

        for metric, values in average_metrics.items():
            value = sum(values) / len(values)
            self.log(metric, value, logger=False)

        self.validation_step_outputs.clear()

        return metrics

    def validate_loss(
        self, dataset_id: str, scores: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        metrics = {}
        if self.loss_functions is None:
            raise ValueError("Loss function is not set")
        for loss_function in self.loss_functions:
            # NOTE skip in-batch losses because they can use a lot of memory
            if isinstance(loss_function, InBatchLossFunction):
                continue
            metrics[f"{dataset_id}/validation-{loss_function.__class__.__name__}"] = (
                loss_function.compute_loss(scores, targets).item()
            )
        return metrics

    def save_pretrained(self, save_path: str | Path) -> None:
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if not self.trainer.training or self.trainer.global_rank != 0:
                return
            _step = self.trainer.global_step
            self.config.save_step = _step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)
