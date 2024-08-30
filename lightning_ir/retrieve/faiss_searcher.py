from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Tuple

import torch

from ..bi_encoder.model import BiEncoderEmbedding
from .searcher import SearchConfig, Searcher

if TYPE_CHECKING:
    from ..bi_encoder import BiEncoderModule


class FaissSearcher(Searcher):
    def __init__(
        self, index_dir: Path | str, search_config: FaissSearchConfig, module: BiEncoderModule, use_gpu: bool = False
    ) -> None:
        import faiss

        self.search_config: FaissSearchConfig
        self.index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
        ivf_index = None
        try:
            ivf_index = faiss.extract_index_ivf(self.index)
        except RuntimeError:
            pass
        if ivf_index is not None:
            ivf_index.nprobe = search_config.n_probe
            quantizer = getattr(ivf_index, "quantizer", None)
            if quantizer is not None:
                downcasted_quantizer = faiss.downcast_index(quantizer)
                hnsw = getattr(downcasted_quantizer, "hnsw", None)
                if hnsw is not None:
                    hnsw.efSearch = search_config.ef_search
        super().__init__(index_dir, search_config, module, use_gpu)

    @property
    def num_embeddings(self) -> int:
        return self.index.ntotal

    @property
    def doc_is_single_vector(self) -> bool:
        return self.num_docs == self.num_embeddings

    def _search(self, query_embeddings: BiEncoderEmbedding) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        query_embeddings = query_embeddings.to(self.device)
        candidate_scores, candidate_doc_idcs = self.candidate_retrieval(query_embeddings)
        query_lengths = query_embeddings.scoring_mask.sum(-1)
        if self.search_config.imputation_strategy == "gather":
            doc_embeddings, doc_idcs, num_docs = self.gather_imputation(candidate_doc_idcs, query_lengths)
            doc_scores = self.module.model.score(query_embeddings, doc_embeddings, num_docs)
        else:
            doc_scores, doc_idcs, num_docs = self.intra_ranking_imputation(
                candidate_scores, candidate_doc_idcs, query_lengths
            )
        return doc_scores, doc_idcs, num_docs

    def candidate_retrieval(self, query_embeddings: BiEncoderEmbedding) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = query_embeddings.embeddings[query_embeddings.scoring_mask]
        candidate_scores, candidate_idcs = self.index.search(embeddings.float().cpu(), self.search_config.candidate_k)
        candidate_scores = torch.from_numpy(candidate_scores)
        candidate_idcs = torch.from_numpy(candidate_idcs)
        if self.doc_is_single_vector:
            candidate_doc_idcs = candidate_idcs.to(self.cumulative_doc_lengths.device)
        else:
            candidate_doc_idcs = torch.searchsorted(
                self.cumulative_doc_lengths,
                candidate_idcs.to(self.cumulative_doc_lengths.device),
                side="right",
            )
        return candidate_scores, candidate_doc_idcs

    def gather_imputation(
        self, candidate_doc_idcs: torch.Tensor, query_lengths: torch.Tensor
    ) -> Tuple[BiEncoderEmbedding, torch.Tensor, List[int]]:
        # unique doc_idcs per query
        doc_idcs_per_query = [
            list(sorted(set(idcs.reshape(-1).tolist())))
            for idcs in torch.split(candidate_doc_idcs, query_lengths.tolist())
        ]
        num_docs = [len(idcs) for idcs in doc_idcs_per_query]
        doc_idcs = torch.tensor(sum(doc_idcs_per_query, [])).to(candidate_doc_idcs)
        unique_doc_idcs, inverse_idcs = torch.unique(doc_idcs, return_inverse=True)

        # gather all vectors for unique doc_idcs
        doc_lengths = self.doc_lengths[unique_doc_idcs]
        start_doc_idcs = self.cumulative_doc_lengths[unique_doc_idcs - 1]
        start_doc_idcs[unique_doc_idcs == 0] = 0
        all_doc_idcs = torch.cat(
            [
                torch.arange(start.item(), start.item() + length.item())
                for start, length in zip(start_doc_idcs.cpu(), doc_lengths.cpu())
            ]
        )
        all_doc_embeddings = torch.from_numpy(self.index.reconstruct_batch(all_doc_idcs))
        unique_embeddings = torch.nn.utils.rnn.pad_sequence(
            [embeddings for embeddings in torch.split(all_doc_embeddings, doc_lengths.tolist())],
            batch_first=True,
        ).to(inverse_idcs.device)
        embeddings = unique_embeddings[inverse_idcs]

        # mask out padding
        doc_lengths = doc_lengths[inverse_idcs]
        scoring_mask = torch.arange(embeddings.shape[1], device=embeddings.device) < doc_lengths[:, None]
        doc_embeddings = BiEncoderEmbedding(embeddings=embeddings, scoring_mask=scoring_mask)
        return doc_embeddings, doc_idcs, num_docs

    def intra_ranking_imputation(
        self,
        candidate_scores: torch.Tensor,
        candidate_doc_idcs: torch.Tensor,
        query_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        max_query_length = int(query_lengths.max().item())
        is_query_single_vector = max_query_length == 1

        if self.doc_is_single_vector:
            scores = candidate_scores.view(-1)
            doc_idcs = candidate_doc_idcs.view(-1)
            num_docs = torch.full((candidate_scores.shape[0],), candidate_scores.shape[1])
        else:
            # grab unique doc ids per query candidate
            query_idcs = torch.arange(query_lengths.shape[0], device=query_lengths.device).repeat_interleave(
                query_lengths
            )
            query_candidate_idcs = torch.cat(
                [torch.arange(length.item(), device=query_lengths.device) for length in query_lengths]
            )
            paired_idcs = torch.stack(
                [
                    query_idcs.repeat_interleave(candidate_scores.shape[1]),
                    query_candidate_idcs.repeat_interleave(candidate_scores.shape[1]),
                    candidate_doc_idcs.view(-1),
                ]
            ).T
            unique_paired_idcs, inverse_idcs = torch.unique(paired_idcs[:, [0, 2]], return_inverse=True, dim=0)
            doc_idcs = unique_paired_idcs[:, 1]
            num_docs = unique_paired_idcs[:, 0].bincount()

            # accumulate max score per doc
            ranking_doc_idcs = torch.arange(doc_idcs.shape[0], device=query_lengths.device)[inverse_idcs]
            idcs = ranking_doc_idcs * max_query_length + paired_idcs[:, 1]
            shape = torch.Size((doc_idcs.shape[0], max_query_length))
            scores = torch.scatter_reduce(
                torch.full((shape.numel(),), float("inf"), device=query_lengths.device),
                0,
                idcs,
                candidate_scores.view(-1).to(query_lengths.device),
                "max",
                include_self=False,
            ).view(shape)

        if is_query_single_vector:
            scores = scores.squeeze(-1)
        else:
            # impute missing values
            if self.search_config.imputation_strategy == "min":
                impute_values = (
                    scores.masked_fill(scores == torch.finfo(scores.dtype).min, float("inf"))
                    .min(0, keepdim=True)
                    .values.expand_as(scores)
                )
            elif self.search_config.imputation_strategy is None:
                impute_values = torch.zeros_like(scores)
            else:
                raise ValueError("Invalid imputation strategy: " f"{self.search_config.imputation_strategy}")
            is_inf = torch.isinf(scores)
            scores[is_inf] = impute_values[is_inf]

            # aggregate score per query vector
            mask = (
                torch.arange(max_query_length, device=query_lengths.device) < query_lengths[:, None]
            ).repeat_interleave(num_docs, dim=0)
            scores = self.module.scoring_function.aggregate(
                scores, mask, self.module.config.query_aggregation_function, dim=1
            ).squeeze(-1)
        return scores, doc_idcs, num_docs.tolist()


class FaissSearchConfig(SearchConfig):
    search_class = FaissSearcher

    def __init__(
        self,
        k: int = 10,
        candidate_k: int = 100,
        imputation_strategy: Literal["min", "gather"] | None = None,
        n_probe: int = 1,
        ef_search: int = 16,
    ) -> None:
        super().__init__(k)
        self.candidate_k = candidate_k
        self.imputation_strategy = imputation_strategy
        self.n_probe = n_probe
        self.ef_search = ef_search
