from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from ..base import LightningIRModule
from ..data import IndexBatch, RankBatch, SearchBatch, TrainBatch
from ..loss.loss import EmbeddingLossFunction, InBatchLossFunction, LossFunction, ScoringLossFunction
from ..retrieve import SearchConfig, Searcher
from .config import BiEncoderConfig
from .model import BiEncoderEmbedding, BiEncoderModel, BiEncoderOutput
from .tokenizer import BiEncoderTokenizer


class BiEncoderModule(LightningIRModule):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: BiEncoderConfig | None = None,
        model: BiEncoderModel | None = None,
        loss_functions: Sequence[LossFunction | Tuple[LossFunction, float]] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        index_dir: Path | None = None,
        search_config: SearchConfig | None = None,
    ):
        super().__init__(model_name_or_path, config, model, loss_functions, evaluation_metrics)
        self.model: BiEncoderModel
        self.config: BiEncoderConfig
        self.tokenizer: BiEncoderTokenizer
        self.scoring_function = self.model.scoring_function
        if self.config.add_marker_tokens and len(self.tokenizer) > self.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), 8)
        self._searcher = None
        self.search_config = search_config
        self.index_dir = index_dir

    @property
    def searcher(self) -> Searcher | None:
        return self._searcher

    @searcher.setter
    def searcher(self, searcher: Searcher):
        self._searcher = searcher

    def on_test_start(self) -> None:
        if self.search_config is not None and self.index_dir is not None:
            self.searcher = self.search_config.search_class(self.index_dir, self.search_config, self)
        return super().on_test_start()

    def forward(self, batch: RankBatch | IndexBatch | SearchBatch) -> BiEncoderOutput:
        queries = getattr(batch, "queries", None)
        docs = getattr(batch, "docs", None)
        num_docs = None
        if isinstance(batch, RankBatch):
            num_docs = None if docs is None else [len(d) for d in docs]
            docs = [d for nested in docs for d in nested] if docs is not None else None
        encodings = self.prepare_input(queries, docs, num_docs)

        if not encodings:
            raise ValueError("No encodings were generated.")
        output = self.model.forward(
            encodings.get("query_encoding", None), encodings.get("doc_encoding", None), num_docs
        )
        if isinstance(batch, SearchBatch) and self.searcher is not None:
            scores, doc_ids, num_docs = self.searcher.search(output)
            output.scores = scores
            cum_num_docs = [0] + [sum(num_docs[: i + 1]) for i in range(len(num_docs))]
            doc_ids = tuple(tuple(doc_ids[cum_num_docs[i] : cum_num_docs[i + 1]]) for i in range(len(num_docs)))
            batch.doc_ids = doc_ids
        return output

    def score(self, queries: Sequence[str] | str, docs: Sequence[Sequence[str]] | Sequence[str]) -> BiEncoderOutput:
        return super().score(queries, docs)

    def compute_losses(self, batch: TrainBatch) -> List[torch.Tensor]:
        if self.loss_functions is None:
            raise ValueError("Loss function is not set")
        output = self.forward(batch)

        scores = output.scores
        query_embeddings = output.query_embeddings
        doc_embeddings = output.doc_embeddings
        if batch.targets is None or query_embeddings is None or doc_embeddings is None or scores is None:
            raise ValueError(
                "targets, scores, query_embeddings, and doc_embeddings must be set in " "the output and batch"
            )

        num_queries = len(batch.queries)
        scores = scores.view(num_queries, -1)
        targets = batch.targets.view(*scores.shape, -1)
        losses = []
        for loss_function, _ in self.loss_functions:
            if isinstance(loss_function, InBatchLossFunction):
                pos_idcs, neg_idcs = loss_function.get_ib_idcs(*scores.shape)
                ib_doc_embeddings = self.get_ib_doc_embeddings(doc_embeddings, pos_idcs, neg_idcs, num_queries)
                ib_scores = self.model.score(query_embeddings, ib_doc_embeddings)
                ib_scores = ib_scores.view(num_queries, -1)
                losses.append(loss_function.compute_loss(ib_scores))
            elif isinstance(loss_function, EmbeddingLossFunction):
                losses.append(loss_function.compute_loss(query_embeddings.embeddings, doc_embeddings.embeddings))
            elif isinstance(loss_function, ScoringLossFunction):
                losses.append(loss_function.compute_loss(scores, targets))
            else:
                raise ValueError(f"Unknown loss function type {loss_function.__class__.__name__}")
        if self.config.sparsification is not None:
            query_num_nonzero = (
                torch.nonzero(query_embeddings.embeddings).shape[0] / query_embeddings.embeddings.shape[0]
            )
            doc_num_nonzero = torch.nonzero(doc_embeddings.embeddings).shape[0] / doc_embeddings.embeddings.shape[0]
            self.log("query_num_nonzero", query_num_nonzero)
            self.log("doc_num_nonzero", doc_num_nonzero)
        return losses

    def get_ib_doc_embeddings(
        self,
        embeddings: BiEncoderEmbedding,
        pos_idcs: torch.Tensor,
        neg_idcs: torch.Tensor,
        num_queries: int,
    ) -> BiEncoderEmbedding:
        _, seq_len, emb_dim = embeddings.embeddings.shape
        ib_embeddings = torch.cat(
            [
                embeddings.embeddings[pos_idcs].view(num_queries, -1, seq_len, emb_dim),
                embeddings.embeddings[neg_idcs].view(num_queries, -1, seq_len, emb_dim),
            ],
            dim=1,
        ).view(-1, seq_len, emb_dim)
        ib_scoring_mask = torch.cat(
            [
                embeddings.scoring_mask[pos_idcs].view(num_queries, -1, seq_len),
                embeddings.scoring_mask[neg_idcs].view(num_queries, -1, seq_len),
            ],
            dim=1,
        ).view(-1, seq_len)
        return BiEncoderEmbedding(ib_embeddings, ib_scoring_mask)

    def validation_step(
        self,
        batch: TrainBatch | IndexBatch | SearchBatch | RankBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> BiEncoderOutput:
        if isinstance(batch, IndexBatch):
            return self.forward(batch)
        if isinstance(batch, (RankBatch, TrainBatch, SearchBatch)):
            return super().validation_step(batch, batch_idx, dataloader_idx)
        raise ValueError(f"Unknown batch type {type(batch)}")
