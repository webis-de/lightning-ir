from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Set, Tuple, Type

import torch

from ...bi_encoder.bi_encoder_model import BiEncoderEmbedding, SingleVectorBiEncoderConfig
from .packed_tensor import PackedTensor

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderModule, BiEncoderOutput


def cat_arange(arange_starts: torch.Tensor, arange_ends: torch.Tensor) -> torch.Tensor:
    arange_lengths = arange_ends - arange_starts
    offsets = torch.cumsum(arange_lengths, dim=0) - arange_lengths - arange_starts
    return torch.arange(arange_lengths.sum()) - torch.repeat_interleave(offsets, arange_lengths)


class Searcher(ABC):
    def __init__(
        self, index_dir: Path | str, search_config: SearchConfig, module: BiEncoderModule, use_gpu: bool = True
    ) -> None:
        super().__init__()
        self.index_dir = Path(index_dir)
        self.search_config = search_config
        self.use_gpu = use_gpu
        self.module = module
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

        self.doc_ids = (self.index_dir / "doc_ids.txt").read_text().split()
        self.doc_lengths = torch.load(self.index_dir / "doc_lengths.pt", weights_only=True)

        self.to_gpu()

        self.num_docs = len(self.doc_ids)
        self.cumulative_doc_lengths = torch.cumsum(self.doc_lengths, dim=0)
        self.num_embeddings = int(self.cumulative_doc_lengths[-1].item())

        self.doc_is_single_vector = self.num_docs == self.num_embeddings
        self.query_is_single_vector = isinstance(module.config, SingleVectorBiEncoderConfig) or getattr(
            module.config, "query_pooling_strategy", None
        ) in {"first", "mean", "min", "max"}

        if self.doc_lengths.shape[0] != self.num_docs or self.doc_lengths.sum() != self.num_embeddings:
            raise ValueError("doc_lengths do not match index")

    def to_gpu(self) -> None:
        self.doc_lengths = self.doc_lengths.to(self.device)

    def _filter_and_sort(
        self, doc_scores: PackedTensor, doc_idcs: PackedTensor, k: int | None = None
    ) -> Tuple[PackedTensor, PackedTensor]:
        k = k or self.search_config.k
        per_query_doc_scores = torch.split(doc_scores, doc_scores.lengths)
        per_query_doc_idcs = torch.split(doc_idcs, doc_idcs.lengths)
        num_docs = []
        new_doc_scores = []
        new_doc_idcs = []
        for _scores, _idcs in zip(per_query_doc_scores, per_query_doc_idcs):
            _k = min(k, _scores.shape[0])
            top_values, top_idcs = torch.topk(_scores, _k)
            new_doc_scores.append(top_values)
            new_doc_idcs.append(_idcs[top_idcs])
            num_docs.append(_k)
        return PackedTensor(torch.cat(new_doc_scores), lengths=num_docs), PackedTensor(
            torch.cat(new_doc_idcs), lengths=num_docs
        )

    @abstractmethod
    def search(self, output: BiEncoderOutput) -> Tuple[PackedTensor, List[List[str]]]: ...


class ExactSearcher(Searcher):

    def search(self, output: BiEncoderOutput) -> Tuple[PackedTensor, List[List[str]]]:
        query_embeddings = output.query_embeddings
        if query_embeddings is None:
            raise ValueError("Expected query_embeddings in BiEncoderOutput")
        query_embeddings = query_embeddings.to(self.device)

        scores = self._score(query_embeddings)

        # aggregate doc token scores
        if not self.doc_is_single_vector:
            scores = torch.scatter_reduce(
                torch.zeros(scores.shape[0], self.num_docs, device=scores.device),
                1,
                self.doc_token_idcs[None].long().expand_as(scores),
                scores,
                "amax",
            )

        # aggregate query token scores
        if not self.query_is_single_vector:
            if query_embeddings.scoring_mask is None:
                raise ValueError("Expected scoring_mask in multi-vector query_embeddings")
            query_lengths = query_embeddings.scoring_mask.sum(-1)
            query_token_idcs = torch.arange(query_lengths.shape[0]).to(query_lengths).repeat_interleave(query_lengths)
            scores = torch.scatter_reduce(
                torch.zeros(query_lengths.shape[0], self.num_docs, device=scores.device),
                0,
                query_token_idcs[:, None].expand_as(scores),
                scores,
                self.module.config.query_aggregation_function,
            )
        top_scores, top_idcs = torch.topk(scores, self.search_config.k)
        doc_ids = [[self.doc_ids[idx] for idx in _doc_idcs] for _doc_idcs in top_idcs.tolist()]
        return PackedTensor(top_scores.view(-1), lengths=[self.search_config.k] * len(doc_ids)), doc_ids

    @property
    def doc_token_idcs(self) -> torch.Tensor:
        if not hasattr(self, "_doc_token_idcs"):
            self._doc_token_idcs = (
                torch.arange(self.doc_lengths.shape[0])
                .to(device=self.doc_lengths.device)
                .repeat_interleave(self.doc_lengths)
            )
        return self._doc_token_idcs

    @abstractmethod
    def _score(self, query_embeddings: BiEncoderEmbedding) -> torch.Tensor: ...


class ApproximateSearcher(Searcher):

    def search(self, output: BiEncoderOutput) -> Tuple[PackedTensor, List[List[str]]]:
        query_embeddings = output.query_embeddings
        if query_embeddings is None:
            raise ValueError("Expected query_embeddings in BiEncoderOutput")
        query_embeddings = query_embeddings.to(self.device)

        candidate_scores, candidate_idcs = self._candidate_retrieval(query_embeddings)
        scores, doc_idcs = self._aggregate_doc_scores(candidate_scores, candidate_idcs, query_embeddings)
        scores = self._aggregate_query_scores(scores, query_embeddings)
        scores, doc_idcs = self._filter_and_sort(scores, doc_idcs)
        doc_ids = [
            [self.doc_ids[doc_idx] for doc_idx in _doc_ids.tolist()] for _doc_ids in doc_idcs.split(doc_idcs.lengths)
        ]

        return scores, doc_ids

    def _aggregate_doc_scores(
        self, candidate_scores: PackedTensor, candidate_idcs: PackedTensor, query_embeddings: BiEncoderEmbedding
    ) -> Tuple[PackedTensor, PackedTensor]:
        if self.doc_is_single_vector:
            return candidate_scores, candidate_idcs

        query_lengths = query_embeddings.scoring_mask.sum(-1)
        num_query_vecs = query_lengths.sum()

        # map vec_idcs to doc_idcs
        candidate_doc_idcs = torch.searchsorted(
            self.cumulative_doc_lengths,
            candidate_idcs.to(self.cumulative_doc_lengths.device),
            side="right",
        )

        # convert candidate_scores `num_query_vecs x candidate_k` to `num_query_doc_pairs x num_query_vecs`
        # and aggregate the maximum doc_vector score per query_vector
        max_query_length = query_lengths.max()
        num_docs_per_query_candidate = torch.tensor(candidate_scores.lengths)

        query_idcs = (
            torch.arange(query_lengths.shape[0], device=query_lengths.device)
            .repeat_interleave(query_lengths)
            .repeat_interleave(num_docs_per_query_candidate)
        )
        query_vector_idcs = cat_arange(torch.zeros_like(query_lengths), query_lengths).repeat_interleave(
            num_docs_per_query_candidate
        )

        stacked = torch.stack([query_idcs, candidate_doc_idcs])
        unique_idcs, ranking_doc_idcs = stacked.unique(return_inverse=True, dim=1)
        num_docs = unique_idcs[0].bincount()
        doc_idcs = PackedTensor(unique_idcs[1], lengths=num_docs.tolist())
        total_num_docs = num_docs.sum()

        unpacked_scores = torch.full((total_num_docs * max_query_length,), float("nan"), device=query_lengths.device)
        index = ranking_doc_idcs * max_query_length + query_vector_idcs
        unpacked_scores = torch.scatter_reduce(
            unpacked_scores, 0, index, candidate_scores, "max", include_self=False
        ).view(total_num_docs, max_query_length)

        # impute the missing values
        if self.search_config.imputation_strategy == "gather":
            # reconstruct the doc embeddings and re-compute the scores
            imputation_values = torch.empty_like(unpacked_scores)
            doc_embeddings = self._reconstruct_doc_embeddings(doc_idcs)
            similarity = self.module.model.compute_similarity(query_embeddings, doc_embeddings, doc_idcs.lengths)
            unpacked_scores = self.module.model._aggregate(
                similarity, doc_embeddings.scoring_mask, "max", dim=-1
            ).squeeze(-1)
        elif self.search_config.imputation_strategy == "min":
            per_query_vec_min = torch.scatter_reduce(
                torch.empty(num_query_vecs),
                0,
                torch.arange(query_lengths.sum()).repeat_interleave(num_docs_per_query_candidate),
                candidate_scores,
                "min",
                include_self=False,
            )
            imputation_values = torch.nn.utils.rnn.pad_sequence(
                per_query_vec_min.split(query_lengths.tolist()), batch_first=True
            ).repeat_interleave(num_docs, dim=0)
        elif self.search_config.imputation_strategy == "zero":
            imputation_values = torch.zeros_like(unpacked_scores)
        else:
            raise ValueError("Invalid imputation strategy: " f"{self.search_config.imputation_strategy}")

        is_nan = torch.isnan(unpacked_scores)
        unpacked_scores[is_nan] = imputation_values[is_nan]

        return PackedTensor(unpacked_scores, lengths=num_docs.tolist()), doc_idcs

    def _aggregate_query_scores(self, scores: PackedTensor, query_embeddings: BiEncoderEmbedding) -> PackedTensor:
        if self.query_is_single_vector:
            return scores
        query_scoring_mask = query_embeddings.scoring_mask.repeat_interleave(torch.tensor(scores.lengths), dim=0)
        scores = PackedTensor(
            self.module.model._aggregate(
                scores, query_scoring_mask, self.module.config.query_aggregation_function, dim=1
            ).squeeze(-1),
            lengths=scores.lengths,
        )
        return scores

    @abstractmethod
    def _candidate_retrieval(self, query_embeddings: BiEncoderEmbedding) -> Tuple[PackedTensor, PackedTensor]:
        """Retrieves initial candidates using the query embeddings. Returns candidate scores and candidate vector
        indices of shape `num_query_vecs x candidate_k` (packed). Candidate indices are None if all doc vectors are
        scored.

        :return: Candidate scores and candidate vector indices
        :rtype: Tuple[PackedTensor, PackedTensor]
        """
        ...

    @abstractmethod
    def _gather_doc_embeddings(self, idcs: torch.Tensor) -> torch.Tensor:
        """Reconstructs embeddings from indices.

        :param doc_idcs: Indices
        :type doc_idcs: PackedTensor
        :return: Reconstructed embeddings
        :rtype: BiEncoderEmbedding
        """
        ...

    def _reconstruct_doc_embeddings(self, doc_idcs: PackedTensor) -> BiEncoderEmbedding:
        # unique doc_idcs per query
        unique_doc_idcs, inverse_idcs = torch.unique(doc_idcs, return_inverse=True)

        # gather all vectors for unique doc_idcs
        doc_lengths = self.doc_lengths[unique_doc_idcs]
        start_doc_idcs = self.cumulative_doc_lengths[unique_doc_idcs - 1]
        start_doc_idcs[unique_doc_idcs == 0] = 0
        all_doc_idcs = cat_arange(start_doc_idcs, start_doc_idcs + doc_lengths)
        all_doc_embeddings = self._gather_doc_embeddings(all_doc_idcs)
        unique_embeddings = torch.nn.utils.rnn.pad_sequence(
            [embeddings for embeddings in torch.split(all_doc_embeddings, doc_lengths.tolist())],
            batch_first=True,
        ).to(inverse_idcs.device)
        embeddings = unique_embeddings[inverse_idcs]

        # mask out padding
        doc_lengths = doc_lengths[inverse_idcs]
        scoring_mask = torch.arange(embeddings.shape[1], device=embeddings.device) < doc_lengths[:, None]
        doc_embeddings = BiEncoderEmbedding(embeddings=embeddings, scoring_mask=scoring_mask, encoding=None)
        return doc_embeddings


class SearchConfig:
    search_class: Type[Searcher]

    SUPPORTED_MODELS: Set[str]

    def __init__(self, k: int = 10) -> None:
        self.k = k


class ExactSearchConfig(SearchConfig):
    search_class = ExactSearcher


class ApproximateSearchConfig(SearchConfig):
    search_class = ApproximateSearcher

    def __init__(
        self, k: int = 10, candidate_k: int = 100, imputation_strategy: Literal["min", "gather", "zero"] = "gather"
    ) -> None:
        super().__init__(k)
        self.k = k
        self.candidate_k = candidate_k
        self.imputation_strategy = imputation_strategy
