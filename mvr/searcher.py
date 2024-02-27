from pathlib import Path
from typing import List, Literal, NamedTuple, Tuple

import scipy.sparse as sp
import faiss
import numpy as np
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
            self.mvr_config.xtr_token_retrieval_k,
        )

        self.index = faiss.read_index(
            str(self.search_config.index_path / "index.faiss")
        )
        self.index.nprobe = self.search_config.n_probe
        self.doc_ids = np.memmap(
            self.search_config.index_path / "doc_ids.npy",
            dtype="S20",
            mode="r",
        )
        self.num_docs = self.doc_ids.shape[0]
        doc_lengths = np.memmap(
            self.search_config.index_path / "doc_lengths.npy",
            dtype="uint16",
            mode="r",
        )
        self.doc_lengths = np.empty_like(doc_lengths)
        self.doc_lengths[:] = doc_lengths[:]
        if (
            self.doc_lengths.shape[0] != self.num_docs
            or self.doc_lengths.sum() != self.index.ntotal
        ):
            raise ValueError("doc_lengths do not match index")
        self.cumulative_doc_lengths = np.cumsum(doc_lengths)

    def score(
        self,
        query_embeddings: np.ndarray,
        query_lengths: np.ndarray,
        doc_embeddings: np.ndarray,
        num_docs: List[int],
    ):
        split_query_embeddings = list(
            torch.split(torch.from_numpy(query_embeddings), query_lengths.tolist())
        )
        torch_query_embeddings = torch.nn.utils.rnn.pad_sequence(
            split_query_embeddings,
            batch_first=True,
            padding_value=self.scoring_function.MASK_VALUE,
        )
        torch_doc_embeddings = torch.from_numpy(doc_embeddings)
        scores = self.scoring_function.score(
            torch_query_embeddings, torch_doc_embeddings, num_docs=num_docs
        )
        scores = scores.numpy()
        scores = scores[scores != self.scoring_function.MASK_VALUE]
        return scores

    def token_retrieval(
        self, query_embeddings: np.ndarray, query_lengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        token_scores, token_idcs = self.index.search(
            query_embeddings, self.search_config.candidate_k
        )
        token_doc_idcs = np.searchsorted(
            self.cumulative_doc_lengths, token_idcs, side="right"
        )
        return token_scores, token_doc_idcs

    def filter_and_sort(
        self, doc_scores: np.ndarray, doc_idcs: np.ndarray, num_docs: List[int]
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        split_idcs = np.cumsum(num_docs[:-1])
        per_query_doc_scores = np.split(doc_scores, split_idcs)
        per_query_doc_idcs = np.split(doc_idcs, split_idcs)
        top_k_idcs = []
        num_retrieved_docs = 0
        new_num_docs = []
        for scores in per_query_doc_scores:
            k = min(self.search_config.k, scores.shape[0])
            top_k_idcs.append(np.argpartition(scores, -k)[-k:][::-1])
            num_retrieved_docs += k
            new_num_docs.append(k)
        doc_scores = np.empty(num_retrieved_docs, dtype=doc_scores.dtype)
        doc_ids = []
        start_idx = 0
        for query_idx, idcs in enumerate(top_k_idcs):
            end_idx = start_idx + len(idcs)
            doc_scores[start_idx:end_idx] = per_query_doc_scores[query_idx][idcs]
            start_idx = end_idx
            ids = self.doc_ids[per_query_doc_idcs[query_idx][idcs]]
            ids = list(map(lambda doc_id: doc_id.decode("utf8").strip(), ids))
            doc_ids.extend(ids)
        return doc_scores, doc_ids, new_num_docs

    def search(
        self, query_embeddings: np.ndarray, query_lengths: np.ndarray
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        token_scores, token_doc_idcs = self.token_retrieval(
            query_embeddings, query_lengths
        )

        if self.search_config.imputation_strategy == "gather":
            doc_embeddings, doc_idcs, num_docs = self._gather_imputation(
                token_doc_idcs, query_lengths
            )
            doc_scores = self.score(
                query_embeddings, query_lengths, doc_embeddings, num_docs
            )
        elif self.search_config.imputation_strategy == "min":
            doc_scores, doc_idcs, num_docs = self._min_imputation(
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

    def _gather_imputation(
        self, token_doc_idcs: np.ndarray, query_lengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        split_doc_idcs = [
            idcs.reshape(-1)
            for idcs in np.split(token_doc_idcs, query_lengths[:-1].cumsum())
        ]
        doc_idcs_per_query = [list(sorted(set(idcs))) for idcs in split_doc_idcs]
        num_docs = [len(idcs) for idcs in doc_idcs_per_query]
        doc_idcs = np.array(sum(doc_idcs_per_query, []))

        unique_doc_idcs, inverse_idcs = np.unique(doc_idcs, return_inverse=True)
        doc_lengths = self.doc_lengths[unique_doc_idcs]
        start_token_idcs = self.cumulative_doc_lengths[unique_doc_idcs - 1]
        start_token_idcs[unique_doc_idcs == 0] = 0
        unique_doc_embeddings = torch.nn.utils.rnn.pad_sequence(
            [
                torch.from_numpy(
                    self.index.reconstruct_n(start_idx.item(), doc_length.item())
                )
                for start_idx, doc_length in zip(start_token_idcs, doc_lengths)
            ],
            batch_first=True,
            padding_value=self.scoring_function.MASK_VALUE,
        )

        doc_embeddings = unique_doc_embeddings[inverse_idcs]
        return doc_embeddings.numpy(), doc_idcs, num_docs

    def _min_imputation(
        self,
        token_scores: np.ndarray,
        token_doc_idcs: np.ndarray,
        query_lengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        # would be nice to vectorize over the query as well
        query_token_scores = np.split(token_scores, query_lengths[:-1].cumsum())
        query_token_doc_idcs = np.split(token_doc_idcs, query_lengths[:-1].cumsum())
        doc_scores = []
        num_docs = []
        doc_idcs = []
        for _token_scores, _token_doc_idcs in zip(
            query_token_scores, query_token_doc_idcs
        ):
            unique_doc_idcs, inverse_doc_idcs = np.unique(
                _token_doc_idcs, return_inverse=True
            )
            doc_token_scores = sparse_doc_aggregation(_token_scores, inverse_doc_idcs)

            is_nan = np.isnan(doc_token_scores)
            doc_token_scores[is_nan] = np.repeat(
                np.nanmin(doc_token_scores, -1), is_nan.sum(-1)
            )

            num_docs.append(doc_token_scores.shape[1])
            doc_scores.append(
                self.scoring_function.aggregate(
                    torch.from_numpy(doc_token_scores.T)
                ).numpy()
            )
            doc_idcs.append(unique_doc_idcs)

        return np.hstack(doc_scores), np.hstack(doc_idcs), num_docs
