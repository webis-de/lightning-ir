from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from lightning import LightningModule
from transformers import PreTrainedModel

from ..data.data import BiEncoderTrainBatch, IndexBatch, SearchBatch
from ..data.datamodule import RunDataset
from ..lightning_utils.validation_utils import (
    create_qrels_from_dicts,
    create_run_from_scores,
    evaluate_run,
)
from ..loss.loss import InBatchLossFunction, LossFunction
from ..tokenizer.tokenizer import BiEncoderTokenizer
from .bi_encoder import BiEncoderConfig, BiEncoderModel


class BiEncoderModule(LightningModule):
    config_class = BiEncoderConfig

    def __init__(
        self,
        model: BiEncoderModel,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__()
        self.model: BiEncoderModel = model
        self.encoder: PreTrainedModel = model.encoder
        self.encoder.embeddings.position_embeddings.requires_grad_(False)
        self.config = self.model.config
        self.loss_functions = loss_functions
        self.evaluation_metrics = evaluation_metrics
        self.tokenizer: BiEncoderTokenizer = BiEncoderTokenizer.from_pretrained(
            self.config.name_or_path, **self.config.to_tokenizer_dict()
        )
        if (
            self.config.add_marker_tokens
            and len(self.tokenizer) != self.config.vocab_size
        ):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer), 8)
        keys = BiEncoderConfig().to_added_args_dict().keys()
        if any(not hasattr(self.config, key) for key in keys):
            raise ValueError(f"Model is missing MVR config attributes {keys}")

        self.validation_step_outputs = defaultdict(lambda: defaultdict(list))

    def on_fit_start(self) -> None:
        self.train()
        return super().on_fit_start()

    def forward(self, batch: BiEncoderTrainBatch) -> torch.Tensor:
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

    def compute_losses(
        self,
        batch: BiEncoderTrainBatch,
        loss_functions: Sequence[LossFunction],
    ) -> Dict[str, torch.Tensor]:
        query_embeddings = self.model.encode_queries(
            batch.query_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.query_encoding.token_type_ids,
        )
        doc_embeddings = self.model.encode_docs(
            batch.doc_encoding.input_ids,
            batch.doc_encoding.attention_mask,
            batch.doc_encoding.token_type_ids,
        )
        query_scoring_mask, doc_scoring_mask = self.model.scoring_masks(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
        )
        scores = self.model.score(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
        )

        num_queries = batch.query_encoding.input_ids.shape[0]
        scores = scores.view(num_queries, -1)
        targets = batch.targets.view(*scores.shape, -1)
        losses = {}
        for loss_function in loss_functions:
            if isinstance(loss_function, InBatchLossFunction):
                pos_mask, neg_mask = loss_function.get_ib_masks(*scores.shape)
                ib_doc_embeddings, ib_scoring_mask = self.get_ib_doc_embeddings(
                    doc_embeddings,
                    doc_scoring_mask,
                    pos_mask,
                    neg_mask,
                    num_queries,
                )
                ib_scores = self.model.score(
                    query_embeddings,
                    ib_doc_embeddings,
                    query_scoring_mask,
                    ib_scoring_mask,
                )
                ib_scores = ib_scores.view(num_queries, -1)
                losses[loss_function.__class__.__name__] = loss_function.compute_loss(
                    ib_scores
                )
            else:
                losses[loss_function.__class__.__name__] = loss_function.compute_loss(
                    scores, targets
                )
        return losses

    def get_ib_doc_embeddings(
        self,
        doc_embeddings: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        num_queries: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_total_docs, seq_len, emb_dim = doc_embeddings.shape
        num_docs = num_total_docs // num_queries
        doc_embeddings = doc_embeddings.repeat(num_queries, 1, 1).view(
            num_queries,
            num_docs * num_queries,
            seq_len,
            emb_dim,
        )
        doc_scoring_mask = doc_scoring_mask.repeat(num_queries, 1).view(
            num_queries, num_docs * num_queries, seq_len
        )
        doc_embeddings = torch.cat(
            [
                doc_embeddings[pos_mask].view(num_queries, -1, seq_len, emb_dim),
                doc_embeddings[neg_mask].view(num_queries, -1, seq_len, emb_dim),
            ],
            dim=1,
        ).view(-1, seq_len, emb_dim)
        doc_scoring_mask = torch.cat(
            [
                doc_scoring_mask[pos_mask].view(num_queries, -1, seq_len),
                doc_scoring_mask[neg_mask].view(num_queries, -1, seq_len),
            ],
            dim=1,
        ).view(-1, seq_len)
        return doc_embeddings, doc_scoring_mask

    def training_step(self, batch: BiEncoderTrainBatch, batch_idx: int) -> torch.Tensor:
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
        batch: BiEncoderTrainBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.evaluation_metrics is None:
            return
        with torch.inference_mode():
            scores = self.forward(batch)

        dataset_id = str(dataloader_idx)
        trainer = None
        try:
            trainer = self.trainer
        except RuntimeError:
            pass

        if trainer is not None and trainer.val_dataloaders is not None:
            dataset = trainer.val_dataloaders[dataloader_idx].dataset
            if not isinstance(dataset, RunDataset):
                raise ValueError(f"Expected a RunDataset for validation, got {dataset}")
            dataset_id = dataset.dataset_id

        self.validation_step_outputs[dataset_id]["scores"].append(scores)
        self.validation_step_outputs[dataset_id]["targets"].append(batch.targets)
        self.validation_step_outputs[dataset_id]["query_ids"].append(batch.query_ids)
        self.validation_step_outputs[dataset_id]["doc_ids"].append(batch.doc_ids)
        self.validation_step_outputs[dataset_id]["qrels"].append(batch.qrels)

    def on_validation_epoch_end(self) -> Dict[str, float] | None:
        if self.evaluation_metrics is None:
            return
        metrics = {}
        for dataset_id in self.validation_step_outputs:
            outputs = self.validation_step_outputs[dataset_id]
            query_ids = sum(outputs["query_ids"], [])
            doc_ids = sum(outputs["doc_ids"], [])
            scores = torch.cat(outputs["scores"]).view(len(query_ids), -1)
            targets = torch.cat(outputs["targets"]).view(*scores.shape, -1)
            try:
                qrels = create_qrels_from_dicts(sum(outputs["qrels"], []))
            except TypeError:
                qrels = None
            run = create_run_from_scores(query_ids, doc_ids, scores)

            if "loss" in self.evaluation_metrics:
                metrics.update(self.validate_loss(dataset_id, scores, targets))

            evaluation_metrics = [
                metric for metric in self.evaluation_metrics if metric != "loss"
            ]
            if evaluation_metrics:
                if qrels is None:
                    raise ValueError("Qrels are not set")
                for metric, value in evaluate_run(
                    run, qrels, evaluation_metrics
                ).items():
                    metrics[f"{dataset_id}/{metric}"] = value

        for key, value in metrics.items():
            self.log(key, value, sync_dist=True)

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
            _step = self.trainer.global_step
            self.config.save_step = _step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
