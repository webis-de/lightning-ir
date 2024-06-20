from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, NamedTuple, Sequence, Tuple

import faiss
import numpy as np
import torch

from ..bi_encoder.model import BiEncoderEmbedding

if TYPE_CHECKING:
    from ..bi_encoder import BiEncoderConfig, BiEncoderModel


class SearchConfig(NamedTuple):
    index_path: Path
    k: int
    candidate_k: int
    imputation_strategy: Literal["min", "gather"]
    n_probe: int = 1


class SparseDocScores:
    def __init__(self, x: np.ndarray, y: np.ndarray, data: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.data = data


class Searcher:
    def __init__(
        self,
        search_config: SearchConfig,
        config: BiEncoderConfig,
        model: BiEncoderModel,
    ) -> None:
        self.search_config = search_config
        self.config = config
        self.model = model

        if self.config.similarity_function == "l2":
            warnings.warn("L2 similarity is not tested and may not work correctly")

        self.index = faiss.read_index(
            str(self.search_config.index_path / "index.faiss")
        )
        self.index.nprobe = self.search_config.n_probe
        self.doc_ids = (
            (self.search_config.index_path / "doc_ids.txt").read_text().split()
        )
        self.doc_lengths = torch.load(self.search_config.index_path / "doc_lengths.pt")
        if torch.cuda.is_available():
            self.doc_lengths = self.doc_lengths.cuda()
        self.num_docs = len(self.doc_ids)
        if (
            self.doc_lengths.shape[0] != self.num_docs
            or self.doc_lengths.sum() != self.index.ntotal
        ):
            raise ValueError("doc_lengths do not match index")
        self.cumulative_doc_lengths = torch.cumsum(self.doc_lengths, dim=0)

    def token_retrieval(
        self, query_embeddings: BiEncoderEmbedding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = query_embeddings.embeddings[query_embeddings.scoring_mask]
        token_scores, token_idcs = self.index.search(
            embeddings.float().cpu(), self.search_config.candidate_k
        )
        token_scores = torch.from_numpy(token_scores)
        token_idcs = torch.from_numpy(token_idcs)
        token_doc_idcs = torch.searchsorted(
            self.cumulative_doc_lengths,
            token_idcs.to(self.cumulative_doc_lengths.device),
            side="right",
        )
        return token_scores, token_doc_idcs

    def filter_and_sort(
        self, doc_scores: torch.Tensor, doc_idcs: torch.Tensor, num_docs: Sequence[int]
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        per_query_doc_scores = torch.split(doc_scores, num_docs)
        per_query_doc_idcs = torch.split(doc_idcs, num_docs)
        new_num_docs = []
        _doc_scores = []
        doc_ids = []
        for query_idx, scores in enumerate(per_query_doc_scores):
            k = min(self.search_config.k, scores.shape[0])
            values, idcs = torch.topk(scores, k)
            _doc_scores.append(values)
            doc_ids.extend(
                [
                    self.doc_ids[doc_idx]
                    for doc_idx in per_query_doc_idcs[query_idx][idcs].cpu()
                ]
            )
            new_num_docs.append(k)
        doc_scores = torch.cat(_doc_scores)
        return doc_scores, doc_ids, new_num_docs

    def search(
        self, query_embeddings: BiEncoderEmbedding
    ) -> Tuple[torch.Tensor, List[str], List[int]]:

        token_scores, token_doc_idcs = self.token_retrieval(query_embeddings)
        query_lengths = query_embeddings.scoring_mask.sum(-1)
        if self.search_config.imputation_strategy == "gather":
            doc_embeddings, doc_idcs, num_docs = self.gather_imputation(
                token_doc_idcs, query_lengths
            )
            doc_scores = self.model.score(query_embeddings, doc_embeddings, num_docs)
        elif self.search_config.imputation_strategy == "min":
            doc_scores, doc_idcs, num_docs = self.min_imputation(
                token_scores, token_doc_idcs, query_lengths
            )
        else:
            raise ValueError(
                f"Unknown imputation strategy {self.search_config.imputation_strategy}"
            )

        doc_scores, doc_ids, num_docs = self.filter_and_sort(
            doc_scores, doc_idcs, num_docs
        )

        return doc_scores, doc_ids, num_docs

    def gather_imputation(
        self, token_doc_idcs: torch.Tensor, query_lengths: torch.Tensor
    ) -> Tuple[BiEncoderEmbedding, torch.Tensor, List[int]]:
        doc_idcs_per_query = [
            list(sorted(set(idcs.reshape(-1).tolist())))
            for idcs in torch.split(token_doc_idcs, query_lengths.tolist())
        ]
        num_docs = [len(idcs) for idcs in doc_idcs_per_query]
        doc_idcs = torch.tensor(sum(doc_idcs_per_query, [])).to(token_doc_idcs)

        unique_doc_idcs, inverse_idcs = torch.unique(doc_idcs, return_inverse=True)
        doc_lengths = self.doc_lengths[unique_doc_idcs]
        start_token_idcs = self.cumulative_doc_lengths[unique_doc_idcs - 1]
        start_token_idcs[unique_doc_idcs == 0] = 0
        token_idcs = torch.cat(
            [
                torch.arange(start.item(), start.item() + length.item())
                for start, length in zip(start_token_idcs.cpu(), doc_lengths.cpu())
            ]
        )
        token_embeddings = torch.from_numpy(self.index.reconstruct_batch(token_idcs))
        unique_embeddings = torch.nn.utils.rnn.pad_sequence(
            [
                embeddings
                for embeddings in torch.split(token_embeddings, doc_lengths.tolist())
            ],
            batch_first=True,
        ).to(inverse_idcs.device)

        embeddings = unique_embeddings[inverse_idcs]
        doc_lengths = doc_lengths[inverse_idcs]
        scoring_mask = (
            torch.arange(embeddings.shape[1], device=embeddings.device)
            < doc_lengths[:, None]
        )
        doc_embeddings = BiEncoderEmbedding(
            embeddings=embeddings, scoring_mask=scoring_mask
        )
        return doc_embeddings, doc_idcs, num_docs

    def min_imputation(
        self,
        token_scores: torch.Tensor,
        token_doc_idcs: torch.Tensor,
        query_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        max_query_length = int(query_lengths.max())

        # grab unique doc ids per query token
        query_idcs = torch.arange(
            query_lengths.shape[0], device=query_lengths.device
        ).repeat_interleave(query_lengths)
        query_token_idcs = torch.cat(
            [
                torch.arange(length.item(), device=query_lengths.device)
                for length in query_lengths
            ]
        )
        paired_idcs = torch.stack(
            [
                query_idcs.repeat_interleave(token_scores.shape[1]),
                query_token_idcs.repeat_interleave(token_scores.shape[1]),
                token_doc_idcs.view(-1),
            ]
        ).T
        unique_paired_idcs, inverse_idcs = torch.unique(
            paired_idcs[:, [0, 2]], return_inverse=True, dim=0
        )
        doc_idcs = unique_paired_idcs[:, 1]
        num_docs = unique_paired_idcs[:, 0].bincount()
        ranking_doc_idcs = torch.arange(doc_idcs.shape[0], device=query_lengths.device)[
            inverse_idcs
        ]

        # accumulate max score per doc
        idcs = ranking_doc_idcs * max_query_length + paired_idcs[:, 1]
        shape = torch.Size((doc_idcs.shape[0], max_query_length))
        scores = torch.scatter_reduce(
            torch.full((shape.numel(),), float("inf"), device=query_lengths.device),
            0,
            idcs,
            token_scores.view(-1).to(query_lengths.device),
            "max",
            include_self=False,
        ).view(shape)

        # impute missing values
        min_values = (
            scores.masked_fill(scores == torch.finfo(scores.dtype).min, float("inf"))
            .min(0)
            .values[None]
            .expand_as(scores)
        )
        is_inf = torch.isinf(scores)
        scores[is_inf] = min_values[is_inf]

        # aggregate score per query token
        mask = (
            torch.arange(max_query_length, device=query_lengths.device)
            < query_lengths[:, None]
        ).repeat_interleave(num_docs, dim=0)
        scores = self.model.scoring_function.aggregate(
            scores, mask, self.config.aggregation_function
        ).squeeze(-1)
        return scores, doc_idcs, num_docs.tolist()
