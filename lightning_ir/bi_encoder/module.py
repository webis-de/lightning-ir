from typing import Dict, Sequence

import torch

from ..base import LightningIRModule
from ..data import BiEncoderRunBatch, IndexBatch, SearchBatch
from ..loss.loss import InBatchLossFunction, LossFunction
from . import BiEncoderConfig, BiEncoderEmbedding, BiEncoderModel, BiEncoderOutput


class BiEncoderModule(LightningIRModule):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: BiEncoderConfig | None = None,
        model: BiEncoderModel | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__(
            model_name_or_path, config, model, loss_functions, evaluation_metrics
        )
        self.model: BiEncoderModel
        self.config: BiEncoderConfig
        if (
            self.config.add_marker_tokens
            and len(self.tokenizer) != self.config.vocab_size
        ):
            self.model.resize_token_embeddings(len(self.tokenizer), 8)
            self.model.resize_token_embeddings(len(self.tokenizer), 8)

    def forward(self, batch: BiEncoderRunBatch) -> BiEncoderOutput:
        num_docs = [len(ids) for ids in batch.doc_ids]
        output = self.model.forward(batch.query_encoding, batch.doc_encoding, num_docs)
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
            batch.query_encoding,
            batch.doc_encoding,
            [len(ids) for ids in batch.doc_ids],
        )

        scores = output.scores
        query_embeddings = output.query_embeddings
        doc_embeddings = output.doc_embeddings
        if (
            batch.targets is None
            or query_embeddings is None
            or doc_embeddings is None
            or scores is None
        ):
            raise ValueError(
                "targets, scores, query_embeddings, and doc_embeddings must be set in "
                "the output and batch"
            )

        num_queries = batch.query_encoding.input_ids.shape[0]
        scores = scores.view(num_queries, -1)
        targets = batch.targets.view(*scores.shape, -1)
        losses = {}
        for loss_function in loss_functions:
            if isinstance(loss_function, InBatchLossFunction):
                pos_mask, neg_mask = loss_function.get_ib_masks(*scores.shape)
                ib_doc_embeddings = self.get_ib_doc_embeddings(
                    doc_embeddings,
                    pos_mask,
                    neg_mask,
                    num_queries,
                )
                ib_scores = self.model.score(query_embeddings, ib_doc_embeddings)
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
        embeddings: BiEncoderEmbedding,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        num_queries: int,
    ) -> BiEncoderEmbedding:
        num_total_docs, seq_len, emb_dim = embeddings.embeddings.shape
        num_docs = num_total_docs // num_queries
        ib_embeddings = embeddings.embeddings.repeat(num_queries, 1, 1).view(
            num_queries,
            num_docs * num_queries,
            seq_len,
            emb_dim,
        )
        ib_scoring_mask = embeddings.scoring_mask.repeat(num_queries, 1).view(
            num_queries, num_docs * num_queries, seq_len
        )
        ib_embeddings = torch.cat(
            [
                ib_embeddings[pos_mask].view(num_queries, -1, seq_len, emb_dim),
                ib_embeddings[neg_mask].view(num_queries, -1, seq_len, emb_dim),
            ],
            dim=1,
        ).view(-1, seq_len, emb_dim)
        ib_scoring_mask = torch.cat(
            [
                ib_scoring_mask[pos_mask].view(num_queries, -1, seq_len),
                ib_scoring_mask[neg_mask].view(num_queries, -1, seq_len),
            ],
            dim=1,
        ).view(-1, seq_len)
        return BiEncoderEmbedding(ib_embeddings, ib_scoring_mask)

    def predict_step(
        self, batch: IndexBatch | SearchBatch | BiEncoderRunBatch, *args, **kwargs
    ) -> BiEncoderOutput:
        if isinstance(batch, IndexBatch):
            return BiEncoderOutput(
                doc_embeddings=self.model.encode_doc(**batch.doc_encoding)
            )
        if isinstance(batch, SearchBatch):
            return BiEncoderOutput(
                query_embeddings=self.model.encode_query(**batch.query_encoding)
            )
        if isinstance(batch, BiEncoderRunBatch):
            return self.forward(batch)
        raise ValueError(f"Unknown batch type {batch.__class__}")
