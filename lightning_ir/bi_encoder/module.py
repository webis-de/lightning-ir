from typing import Any, Dict, Sequence, Tuple

import torch

from ..data.data import BiEncoderRunBatch, IndexBatch, SearchBatch
from ..loss.loss import InBatchLossFunction, LossFunction
from ..module import LightningIRModule
from .model import BiEncoderConfig, BiEncoderModel, BiEncoderOutput


class BiEncoderModule(LightningIRModule):
    config_class = BiEncoderConfig

    def __init__(
        self,
        model: BiEncoderModel,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__(model, loss_functions, evaluation_metrics)
        if (
            self.config.add_marker_tokens
            and len(self.tokenizer) != self.config.vocab_size
        ):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer), 8)

    def forward(self, batch: BiEncoderRunBatch) -> BiEncoderOutput:
        num_docs = [len(ids) for ids in batch.doc_ids]
        output = self.model.forward(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
            batch.query_encoding.token_type_ids,
            batch.doc_encoding.token_type_ids,
            num_docs,
        )
        return output

    def compute_losses(
        self,
        batch: BiEncoderRunBatch,
        loss_functions: Sequence[LossFunction] | None,
    ) -> Dict[str, torch.Tensor]:
        if loss_functions is None:
            if self.loss_functions is None:
                raise ValueError("Loss function is not set")
            loss_functions = self.loss_functions
        output = self.model.forward(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
            batch.query_encoding.token_type_ids,
            batch.doc_encoding.token_type_ids,
            [len(ids) for ids in batch.doc_ids],
        )
        query_scoring_mask, doc_scoring_mask = self.model.scoring_masks(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
        )

        scores = output.scores
        query_embeddings = output.query_embeddings
        doc_embeddings = output.doc_embeddings

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

    def predict_step(
        self, batch: IndexBatch | SearchBatch | BiEncoderRunBatch, *args, **kwargs
    ) -> BiEncoderOutput:
        if isinstance(batch, IndexBatch):
            return BiEncoderOutput(
                doc_embeddings=self.model.encode_docs(**batch.doc_encoding)
            )
        if isinstance(batch, SearchBatch):
            return BiEncoderOutput(
                query_embeddings=self.model.encode_queries(**batch.query_encoding)
            )
        if isinstance(batch, BiEncoderRunBatch):
            return self.forward(batch)
        raise ValueError(f"Unknown batch type {type(batch)}")
