from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import torch
from lightning import LightningModule
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from transformers import PreTrainedModel

from .data import IndexBatch, SearchBatch, TrainBatch
from .loss import LossFunction
from .mvr import MVRConfig, MVRModel
from .tokenizer import MVRTokenizer


class MVRModule(LightningModule):
    def __init__(self, model: MVRModel, loss_function: LossFunction | None = None):
        super().__init__()
        self.model: MVRModel = model
        self.encoder: PreTrainedModel = model.encoder
        self.encoder.embeddings.position_embeddings.requires_grad_(False)
        self.config = self.model.config
        self.loss_function: LossFunction | None = loss_function
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

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        if self.loss_function is None:
            raise ValueError("Loss function is not set")
        query_embeddings = self.model.encode_queries(**batch.query_encoding)
        doc_embeddings = self.model.encode_docs(**batch.doc_encoding)
        query_scoring_mask, doc_scoring_mask = self.model.scoring_masks(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
        )
        losses = self.loss_function.compute_loss(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
            batch.targets,
        )
        for key, loss in losses.items():
            self.log(key, loss)
        loss = sum(losses.values(), torch.tensor(0))
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: TrainBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        scores = self.forward(batch)
        scores = scores.view(batch.query_encoding.input_ids.shape[0], -1)
        depth = scores.shape[-1]
        relevances = batch.relevances
        assert relevances is not None
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
            aggregated[key].extend(value)

        self.validation_step_outputs.clear()

        for key, value in aggregated.items():
            stacked = torch.stack(value)
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
