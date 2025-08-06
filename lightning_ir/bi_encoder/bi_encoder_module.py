"""
Module module for bi-encoder models.

This module defines the Lightning IR module class used to implement bi-encoder models.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple

import torch
from transformers import BatchEncoding

from ..base import LightningIRModule, LightningIROutput
from ..data import IndexBatch, RankBatch, SearchBatch, TrainBatch
from ..loss.base import EmbeddingLossFunction, LossFunction, ScoringLossFunction
from ..loss.in_batch import InBatchLossFunction
from .bi_encoder_config import BiEncoderConfig
from .bi_encoder_model import BiEncoderEmbedding, BiEncoderModel, BiEncoderOutput
from .bi_encoder_tokenizer import BiEncoderTokenizer

if TYPE_CHECKING:
    from ..retrieve import SearchConfig, Searcher


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
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        """:class:`.LightningIRModule` for bi-encoder models. It contains a :class:`.BiEncoderModel` and a
        :class:`.BiEncoderTokenizer` and implements the training, validation, and testing steps for the model.

        .. _ir-measures: https://ir-measur.es/en/latest/index.html

        Args:
            model_name_or_path (str | None): Name or path of backbone model or fine-tuned Lightning IR model.
                Defaults to None.
            config (BiEncoderConfig | None): BiEncoderConfig to apply when loading from backbone model.
                Defaults to None.
            model (BiEncoderModel | None): Already instantiated BiEncoderModel. Defaults to None.
            loss_functions (Sequence[LossFunction | Tuple[LossFunction, float]] | None):
                Loss functions to apply during fine-tuning, optional loss weights can be provided per loss function
                Defaults to None.
            evaluation_metrics (Sequence[str] | None): Metrics corresponding to ir-measures_ measure strings
                to apply during validation or testing. Defaults to None.
            index_dir (Path | None): Path to an index used for retrieval. Defaults to None.
            search_config (SearchConfig | None): Configuration to use during retrieval. Defaults to None.
            model_kwargs (Mapping[str, Any] | None): Additional keyword arguments to pass to `from_pretrained`
                when loading a model. Defaults to None.
        """
        super().__init__(model_name_or_path, config, model, loss_functions, evaluation_metrics, model_kwargs)
        self.model: BiEncoderModel
        self.config: BiEncoderConfig
        self.tokenizer: BiEncoderTokenizer
        if self.config.add_marker_tokens and len(self.tokenizer) > self.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), 8)
        self._searcher = None
        self.search_config = search_config
        self.index_dir = index_dir

    @property
    def searcher(self) -> Searcher | None:
        """Searcher used for retrieval if `index_dir` and `search_config` are set.

        Returns:
            Searcher: Searcher class.
        """
        return self._searcher

    @searcher.setter
    def searcher(self, searcher: Searcher):
        self._searcher = searcher

    def _init_searcher(self) -> None:
        if self.search_config is not None and self.index_dir is not None:
            self.searcher = self.search_config.search_class(self.index_dir, self.search_config, self)

    def on_test_start(self) -> None:
        """Called at the beginning of testing. Initializes the searcher if `index_dir` and `search_config` are set."""
        self._init_searcher()
        return super().on_test_start()

    def forward(self, batch: RankBatch | IndexBatch | SearchBatch) -> BiEncoderOutput:
        """Runs a forward pass of the model on a batch of data. The output will vary depending on the type of batch. If
        the batch is a :class`.RankBatch`, query and document embeddings are computed and the relevance score is the
        similarity between the two embeddings. If the batch is an :class:`.IndexBatch`, only document embeddings
        are comuputed. If the batch is a :class:`.SearchBatch`, only query embeddings are computed and
        the model will additionally retrieve documents if :attr:`.searcher` is set.

        Args:
            batch (RankBatch | IndexBatch | SearchBatch): Input batch containing queries and/or documents.
        Returns:
            BiEncoderOutput: Output of the model.
        Raises:
            ValueError: If the input batch contains neither queries nor documents.
        """
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
        doc_ids = getattr(batch, "doc_ids", None)
        if doc_ids is not None and output.doc_embeddings is not None:
            output.doc_embeddings.ids = doc_ids
        query_ids = getattr(batch, "query_ids", None)
        if query_ids is not None and output.query_embeddings is not None:
            output.query_embeddings.ids = query_ids
        if isinstance(batch, SearchBatch) and self.searcher is not None:
            scores, doc_ids = self.searcher.search(output)
            output.scores = scores
            if output.doc_embeddings is not None:
                output.doc_embeddings.ids = [doc_id for _doc_ids in doc_ids for doc_id in _doc_ids]
            batch.doc_ids = doc_ids
        return output

    def score(self, queries: Sequence[str] | str, docs: Sequence[Sequence[str]] | Sequence[str]) -> BiEncoderOutput:
        """Computes relevance scores for queries and documents.

        Args:
            queries (Sequence[str] | str): Queries to score.
            docs (Sequence[Sequence[str]] | Sequence[str]): Documents to score.
        Returns:
            BiEncoderOutput: Output of the model.
        """
        return super().score(queries, docs)

    def _compute_losses(self, batch: TrainBatch, output: BiEncoderOutput) -> List[torch.Tensor]:
        """Computes the losses for a training batch."""
        if self.loss_functions is None:
            raise ValueError("Loss function is not set")

        if (
            batch.targets is None
            or output.query_embeddings is None
            or output.doc_embeddings is None
            or output.scores is None
        ):
            raise ValueError(
                "targets, scores, query_embeddings, and doc_embeddings must be set in " "the output and batch"
            )

        num_queries = len(batch.queries)
        output.scores = output.scores.view(num_queries, -1)
        batch.targets = batch.targets.view(*output.scores.shape, -1)
        losses = []
        for loss_function, _ in self.loss_functions:
            if isinstance(loss_function, InBatchLossFunction):
                pos_idcs, neg_idcs = loss_function.get_ib_idcs(output, batch)
                ib_doc_embeddings = self._get_ib_doc_embeddings(output.doc_embeddings, pos_idcs, neg_idcs, num_queries)
                ib_scores = self.model.score(output.query_embeddings, ib_doc_embeddings)
                ib_scores = ib_scores.view(num_queries, -1)
                losses.append(loss_function.compute_loss(LightningIROutput(ib_scores)))
            elif isinstance(loss_function, EmbeddingLossFunction):
                losses.append(loss_function.compute_loss(output))
            elif isinstance(loss_function, ScoringLossFunction):
                losses.append(loss_function.compute_loss(output, batch))
            else:
                raise ValueError(f"Unknown loss function type {loss_function.__class__.__name__}")
        if self.config.sparsification is not None:
            query_num_nonzero = (
                torch.nonzero(output.query_embeddings.embeddings).shape[0] / output.query_embeddings.embeddings.shape[0]
            )
            doc_num_nonzero = (
                torch.nonzero(output.doc_embeddings.embeddings).shape[0] / output.doc_embeddings.embeddings.shape[0]
            )
            self.log("query_num_nonzero", query_num_nonzero)
            self.log("doc_num_nonzero", doc_num_nonzero)
        return losses

    def _get_ib_doc_embeddings(
        self,
        embeddings: BiEncoderEmbedding,
        pos_idcs: torch.Tensor,
        neg_idcs: torch.Tensor,
        num_queries: int,
    ) -> BiEncoderEmbedding:
        """Gets the in-batch document embeddings for a training batch."""
        _, num_embs, emb_dim = embeddings.embeddings.shape
        ib_embeddings = torch.cat(
            [
                embeddings.embeddings[pos_idcs].view(num_queries, -1, num_embs, emb_dim),
                embeddings.embeddings[neg_idcs].view(num_queries, -1, num_embs, emb_dim),
            ],
            dim=1,
        ).view(-1, num_embs, emb_dim)
        if embeddings.scoring_mask is None:
            ib_scoring_mask = None
        else:
            ib_scoring_mask = torch.cat(
                [
                    embeddings.scoring_mask[pos_idcs].view(num_queries, -1, num_embs),
                    embeddings.scoring_mask[neg_idcs].view(num_queries, -1, num_embs),
                ],
                dim=1,
            ).view(-1, num_embs)
        if embeddings.encoding is None:
            ib_encoding = None
        else:
            ib_encoding = {}
            for key, value in embeddings.encoding.items():
                seq_len = value.shape[-1]
                ib_encoding[key] = torch.cat(
                    [value[pos_idcs].view(num_queries, -1, seq_len), value[neg_idcs].view(num_queries, -1, seq_len)],
                    dim=1,
                ).view(-1, seq_len)
            ib_encoding = BatchEncoding(ib_encoding)
        return BiEncoderEmbedding(ib_embeddings, ib_scoring_mask, ib_encoding)

    def validation_step(
        self,
        batch: TrainBatch | IndexBatch | SearchBatch | RankBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> BiEncoderOutput:
        """Handles the validation step for the model.

        Args:
            batch (TrainBatch | IndexBatch | SearchBatch | RankBatch): Batch of validation or testing data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        Returns:
            BiEncoderOutput: Output of the model.
        """
        if isinstance(batch, IndexBatch):
            return self.forward(batch)
        if isinstance(batch, (RankBatch, TrainBatch, SearchBatch)):
            return super().validation_step(batch, batch_idx, dataloader_idx)
        raise ValueError(f"Unknown batch type {type(batch)}")
