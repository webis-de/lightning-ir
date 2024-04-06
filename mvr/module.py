from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from lightning import LightningModule
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from transformers import PreTrainedModel

from .data import IndexBatch, SearchBatch, TrainBatch
from .loss import LossFunction, SimilarityLossFunction
from .mvr import MVRConfig, MVRModel
from .tokenizer import MVRTokenizer


class MVRModule(LightningModule):
    def __init__(
        self, model: MVRModel, loss_functions: Sequence[LossFunction] | None = None
    ):
        super().__init__()
        self.model: MVRModel = model
        self.encoder: PreTrainedModel = model.encoder
        self.encoder.embeddings.position_embeddings.requires_grad_(False)
        self.config = self.model.config
        if loss_functions is not None:
            loss_functions = list(loss_functions)
            for loss_function in loss_functions:
                loss_function.set_scoring_function(self.model.scoring_function)
        self.loss_functions = loss_functions
        self.tokenizer: MVRTokenizer = MVRTokenizer.from_pretrained(
            self.config.name_or_path, **self.config.to_tokenizer_dict()
        )
        if (
            self.config.add_marker_tokens
            and len(self.tokenizer) != self.config.vocab_size
        ):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer), 8)
        keys = MVRConfig().to_mvr_dict().keys()
        if any(not hasattr(self.config, key) for key in keys):
            raise ValueError(f"Model is missing MVR config attributes {keys}")

        self.validation_step_outputs = []

    def on_fit_start(self) -> None:
        self.train()
        return super().on_fit_start()

    def forward(self, batch: TrainBatch) -> torch.Tensor:
        num_docs = [len(ids) for ids in batch.doc_ids]
        scores = self.model.forward(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
            batch.query_encoding.token_type_ids,
            batch.doc_encoding.token_type_ids,
            num_docs,
        )
        return scores

    def step(
        self, batch: TrainBatch, loss_functions: Sequence[LossFunction]
    ) -> Dict[str, torch.Tensor]:
        query_embeddings = self.model.encode_queries(**batch.query_encoding)
        doc_embeddings = self.model.encode_docs(**batch.doc_encoding)
        query_scoring_mask, doc_scoring_mask = self.model.scoring_masks(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
        )
        losses = {}
        for loss_function in loss_functions:
            losses = {
                **losses,
                **loss_function.compute_loss(
                    query_embeddings,
                    doc_embeddings,
                    query_scoring_mask,
                    doc_scoring_mask,
                    batch.targets,
                ),
            }
        return losses

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        if self.loss_functions is None:
            raise ValueError("Loss function is not set")
        losses = self.step(batch, self.loss_functions)
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
        if batch.relevances is None:
            self.tuples_validation_step(batch)
        else:
            self.run_validation_step(batch, dataloader_idx)

    def tuples_validation_step(self, batch: TrainBatch) -> None:
        if self.loss_functions is None:
            raise ValueError("Loss function is not set")
        loss_functions = [
            loss_function
            for loss_function in self.loss_functions
            if isinstance(loss_function, SimilarityLossFunction)
        ]
        losses = self.step(batch, loss_functions)
        self.validation_step_outputs.extend(
            (f"validation {key}", value) for key, value in losses.items()
        )

    def run_validation_step(self, batch: TrainBatch, dataloader_idx: int) -> None:
        relevances = batch.relevances
        if relevances is None:
            raise ValueError("Relevances are required for validation")
        scores = self.forward(batch)
        scores = scores.view(batch.query_encoding.input_ids.shape[0], -1)
        depth = scores.shape[-1]
        scores = torch.nn.functional.pad(
            scores, (0, relevances.shape[-1] - scores.shape[-1])
        )
        dataset_name = ""
        first_stage = ""
        try:
            dataset_path = Path(
                self.trainer.datamodule.inference_datasets[dataloader_idx]
            )
            dataset_name = dataset_path.stem + "-"
            first_stage = dataset_path.parent.name + "-"
        except RuntimeError:
            pass

        metrics = (
            RetrievalNormalizedDCG(top_k=10, aggregation=lambda x, dim: x),
            RetrievalMRR(top_k=depth, aggregation=lambda x, dim: x),
        )
        for metric_name, metric in zip(("ndcg@10", "mrr@ranking"), (metrics)):
            value = metric(
                scores,
                relevances.clamp(0, 1) if "mrr" in metric_name else relevances,
                torch.arange(scores.shape[0])[:, None].expand_as(scores),
            )
            self.validation_step_outputs.append(
                (f"{first_stage}{dataset_name}{metric_name}", value)
            )

    def on_validation_epoch_end(self) -> None:
        aggregated = defaultdict(list)
        for key, value in self.validation_step_outputs:
            aggregated[key].append(value)

        self.validation_step_outputs.clear()

        for key, value in aggregated.items():
            stacked = torch.cat(value).view(-1)
            stacked[torch.isnan(stacked)] = 0
            self.log(key, stacked.mean(), sync_dist=True)

    def predict_step(self, batch: IndexBatch | SearchBatch, *args, **kwargs) -> Any:
        if isinstance(batch, IndexBatch):
            return self.model.encode_docs(**batch.doc_encoding)
        if isinstance(batch, SearchBatch):
            return self.model.encode_queries(**batch.query_encoding)
        raise ValueError(f"Unknown batch type {type(batch)}")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if not self.trainer.training or self.trainer.global_rank != 0:
                return
            step = self.trainer.global_step
            self.config.save_step = step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
