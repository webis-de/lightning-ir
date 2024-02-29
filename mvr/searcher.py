import warnings
from pathlib import Path
from typing import List, Literal, NamedTuple, Tuple

import faiss
import numpy as np
import scipy.sparse as sp
import torch

from .mvr import MVRConfig, ScoringFunction


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


def sparse_doc_aggregation(
    token_scores: np.ndarray, doc_idcs: np.ndarray
) -> np.ndarray:
    # doc aggregation step:
    # need to handle the special case where tokens from the same doc are retrieved
    # by the same query. we need to efficiently aggregate these tokens scores.
    # solution: we can simply invert the idcs and scores for each doc and use the
    # scipy sparse dok matrix to handle duplicate index assignment.
    # scores are already sorted by faiss. we can simply use the last score as
    # the maximum (ip) or minimum (l2) score for each doc.
    # https://stackoverflow.com/questions/40723534/overwrite-instead-of-add-for-duplicate-triplets-when-creating-sparse-matrix-in-scipy
    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
    num_tokens, k = token_scores.shape
    query_token_idcs = np.arange(num_tokens).repeat(k)
    doc_scores = sp.dok_matrix(
        (num_tokens, doc_idcs.max() + 1), dtype=token_scores.dtype
    )
    inv_query_token_idcs = query_token_idcs[::-1]
    inv_doc_idcs = doc_idcs.flatten()[::-1]
    inv_token_scores = token_scores.flatten()[::-1]
    doc_scores._update(zip(zip(inv_query_token_idcs, inv_doc_idcs), inv_token_scores))
    doc_scores = doc_scores.toarray()
    doc_scores[doc_scores == 0] = np.nan
    return doc_scores


class Searcher:
    def __init__(self, search_config: SearchConfig, mvr_config: MVRConfig) -> None:
        self.search_config = search_config
        self.mvr_config = mvr_config
        self.scoring_function = ScoringFunction(
            self.mvr_config.similarity_function,
            self.mvr_config.aggregation_function,
        )

        if self.mvr_config.similarity_function == "l2":
            warnings.warn("L2 similarity is not tested and may not work correctly")

        self.index = faiss.read_index(
            str(self.search_config.index_path / "index.faiss")
        )
        self.index.nprobe = self.search_config.n_probe
        self.doc_ids = torch.load(self.search_config.index_path / "doc_ids.pt")
        self.doc_ids = self.doc_ids.view(-1, 20)
        self.doc_lengths = torch.load(self.search_config.index_path / "doc_lengths.pt")
        self.num_docs = self.doc_ids.shape[0]
        if (
            self.doc_lengths.shape[0] != self.num_docs
            or self.doc_lengths.sum() != self.index.ntotal
        ):
            raise ValueError("doc_lengths do not match index")
        self.cumulative_doc_lengths = torch.cumsum(self.doc_lengths, dim=0)

    def score(
        self,
        query_embeddings: np.ndarray,
        query_lengths: np.ndarray,
        doc_embeddings: np.ndarray,
        num_docs: List[int],
    ):

        torch_doc_embeddings = torch.from_numpy(doc_embeddings)
        scores = self.scoring_function.score(
            torch_query_embeddings, torch_doc_embeddings, num_docs=num_docs
        )
        return scores

    def token_retrieval(
        self, query_embeddings: torch.Tensor, query_scoring_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_embeddings = query_embeddings[query_scoring_mask]
        token_scores, token_idcs = self.index.search(
            query_embeddings.cpu(), self.search_config.candidate_k
        )
        token_scores = torch.from_numpy(token_scores).to(query_embeddings.device)
        token_idcs = torch.from_numpy(token_idcs).to(query_embeddings.device)
        token_doc_idcs = torch.searchsorted(
            self.cumulative_doc_lengths, token_idcs, side="right"
        )
        return token_scores, token_doc_idcs

    def filter_and_sort(
        self, doc_scores: torch.Tensor, doc_idcs: torch.Tensor, num_docs: List[int]
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
            ids = self.doc_ids[per_query_doc_idcs[query_idx][idcs]]
            doc_ids.extend(map(lambda x: bytes(x).decode("utf-8").strip(), ids))
            new_num_docs.append(k)
        doc_scores = torch.cat(_doc_scores)
        return doc_scores, doc_ids, new_num_docs

    def search(
        self, query_embeddings: torch.Tensor, query_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        query_scoring_mask = (
            torch.arange(query_embeddings.shape[1]) < query_lengths[:, None]
        )
        token_scores, token_doc_idcs = self.token_retrieval(
            query_embeddings, query_scoring_mask
        )
        if self.search_config.imputation_strategy == "gather":
            doc_embeddings, doc_idcs, doc_lengths, num_docs = self.gather_imputation(
                token_doc_idcs, query_lengths
            )
            doc_scoring_mask = (
                torch.arange(doc_embeddings.shape[1]) < doc_lengths[:, None]
            )
            doc_scores = self.scoring_function.score(
                query_embeddings,
                doc_embeddings,
                query_scoring_mask,
                doc_scoring_mask,
                num_docs,
            )
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        doc_idcs_per_query = [
            list(sorted(set(idcs.reshape(-1).cpu().tolist())))
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
                for start, length in zip(start_token_idcs, doc_lengths)
            ]
        )
        doc_token_embeddings = self.index.reconstruct_batch(token_idcs)
        doc_token_embeddings = torch.from_numpy(doc_token_embeddings)
        unique_doc_embeddings = torch.nn.utils.rnn.pad_sequence(
            [
                embeddings
                for embeddings in torch.split(
                    doc_token_embeddings, doc_lengths.tolist()
                )
            ],
            batch_first=True,
            padding_value=self.scoring_function.MASK_VALUE,
        )

        doc_embeddings = unique_doc_embeddings[inverse_idcs].to(token_doc_idcs.device)
        doc_lengths = doc_lengths[inverse_idcs]
        return doc_embeddings, doc_idcs, doc_lengths, num_docs

    def min_imputation(
        self,
        token_scores: torch.Tensor,
        token_doc_idcs: torch.Tensor,
        query_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        max_query_length = int(query_lengths.max())

        # grab unique doc ids per query token
        query_idcs = torch.arange(query_lengths.shape[0]).repeat_interleave(
            query_lengths
        )
        query_token_idcs = torch.cat(
            [torch.arange(length.item()) for length in query_lengths]
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
        ranking_doc_idcs = torch.arange(doc_idcs.shape[0])[inverse_idcs]

        # accumulate max score per doc
        idcs = ranking_doc_idcs * max_query_length + paired_idcs[:, 1]
        shape = torch.Size((doc_idcs.shape[0], max_query_length))
        scores = torch.scatter_reduce(
            torch.full((shape.numel(),), float("inf")),
            0,
            idcs,
            token_scores.view(-1),
            "max",
            include_self=False,
        ).view(shape)

        # impute missing values
        min_values = scores.min(0).values[None].expand_as(scores)
        is_inf = torch.isinf(scores)
        scores[is_inf] = min_values[is_inf]

        # aggregate score per query token
        mask = (
            torch.arange(max_query_length) < query_lengths[:, None]
        ).repeat_interleave(num_docs, dim=0)
        flat_scores = scores[mask]
        scores = self.scoring_function.aggregate(
            flat_scores, mask, self.mvr_config.aggregation_function
        )
        return scores, doc_idcs, num_docs.tolist()
