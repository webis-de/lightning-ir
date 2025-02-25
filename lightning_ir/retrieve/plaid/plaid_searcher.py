from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Type

import torch

from ...bi_encoder.model import BiEncoderEmbedding
from ..base.searcher import SearchConfig, Searcher
from .packed_tensor import PackedTensor
from .plaid_indexer import PlaidIndexConfig
from .residual_codec import ResidualCodec

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderModule


class PlaidSearcher(Searcher):
    def __init__(
        self, index_dir: Path | str, search_config: PlaidSearchConfig, module: BiEncoderModule, use_gpu: bool = False
    ) -> None:
        super().__init__(index_dir, search_config, module, use_gpu)
        self.residual_codec = ResidualCodec.from_pretrained(
            PlaidIndexConfig.from_pretrained(self.index_dir), self.index_dir
        )

        self.codes = torch.load(self.index_dir / "codes.pt")
        self.residuals = torch.load(self.index_dir / "residuals.pt").view(self.codes.shape[0], -1)
        self.packed_codes = PackedTensor(self.codes, self.doc_lengths.tolist())
        self.packed_residuals = PackedTensor(self.residuals, self.doc_lengths.tolist())

        # code_idx to embedding_idcs mapping
        sorted_codes, embedding_idcs = self.codes.sort()
        num_embeddings_per_code = torch.bincount(sorted_codes, minlength=self.residual_codec.num_centroids).tolist()
        # self.code_to_embedding_ivf = PackedTensor(embedding_idcs, num_embeddings_per_code)

        # code_idx to doc_idcs mapping
        embedding_idx_to_doc_idx = torch.arange(self.num_docs).repeat_interleave(self.doc_lengths)
        full_doc_ivf = embedding_idx_to_doc_idx[embedding_idcs]
        doc_ivf_lengths = []
        unique_doc_idcs = []
        for doc_idcs in full_doc_ivf.split(num_embeddings_per_code):
            unique_doc_idcs.append(doc_idcs.unique())
            doc_ivf_lengths.append(unique_doc_idcs[-1].shape[0])
        self.code_to_doc_ivf = PackedTensor(torch.cat(unique_doc_idcs), doc_ivf_lengths)

        # doc_idx to code_idcs mapping
        sorted_doc_idcs, doc_idx_to_code_idx = self.code_to_doc_ivf.packed_tensor.sort()
        code_idcs = torch.arange(self.residual_codec.num_centroids).repeat_interleave(
            torch.tensor(self.code_to_doc_ivf.lengths)
        )[doc_idx_to_code_idx]
        num_codes_per_doc = torch.bincount(sorted_doc_idcs, minlength=self.num_docs)
        self.doc_to_code_ivf = PackedTensor(code_idcs, num_codes_per_doc.tolist())

        self.search_config: PlaidSearchConfig

    @property
    def num_embeddings(self) -> int:
        return int(self.cumulative_doc_lengths[-1].item())

    def candidate_retrieval(self, query_embeddings: BiEncoderEmbedding) -> Tuple[torch.Tensor, PackedTensor]:
        # grab top `n_cells` neighbor cells for all query embeddings
        # `num_queries x query_length x num_centroids`
        scores = (
            query_embeddings.embeddings.to(self.residual_codec.centroids)
            @ self.residual_codec.centroids.transpose(0, 1)[None]
        )
        scores = scores.masked_fill(~query_embeddings.scoring_mask[..., None], 0)
        _, codes = torch.topk(scores, self.search_config.n_cells, dim=-1, sorted=False)
        packed_codes = codes[query_embeddings.scoring_mask].view(-1)
        code_lengths = (query_embeddings.scoring_mask.sum(-1) * self.search_config.n_cells).tolist()

        # grab document idcs for all cells
        packed_doc_idcs = self.code_to_doc_ivf.lookup(packed_codes, code_lengths, unique=True)
        return scores, packed_doc_idcs

    def filter_candidates(
        self, centroid_scores: torch.Tensor, doc_idcs: PackedTensor, threshold: float | None, k: int
    ) -> PackedTensor:
        num_query_vecs = centroid_scores.shape[1]
        num_centroids = centroid_scores.shape[-1]

        # repeat query centroid scores for each document
        # `num_docs x num_query_vecs x num_centroids + 1`
        # NOTE we pad values such that the codes with -1 padding index 0 values
        expanded_centroid_scores = torch.nn.functional.pad(
            centroid_scores.repeat_interleave(torch.tensor(doc_idcs.lengths), dim=0), (0, 1)
        )

        # grab codes for each document
        code_idcs = self.doc_to_code_ivf.lookup(doc_idcs.packed_tensor, 1)
        # `num_docs x max_num_codes_per_doc`
        padded_codes = code_idcs.to_padded_tensor(pad_value=num_centroids)
        mask = padded_codes != num_centroids
        # `num_docs x max_num_query_vecs x max_num_codes_per_doc`
        padded_codes = padded_codes[:, None].expand(-1, num_query_vecs, -1)

        # apply pruning threshold
        if threshold is not None and threshold:
            expanded_centroid_scores = expanded_centroid_scores.masked_fill(
                expanded_centroid_scores.amax(1, keepdim=True) < threshold, 0
            )

        # NOTE this is colbert scoring, but instead of using the doc embeddings we use the centroid scores
        # expanded_centroid_scores: `num_docs x max_num_query_vecs x num_centroids + 1 `
        # padded_codes: `num_docs x max_num_query_vecs x max_num_codes_per_doc`
        # approx_similarity: `num_docs x max_num_query_vecs x max_num_codes_per_doc`
        approx_similarity = torch.gather(input=expanded_centroid_scores, dim=-1, index=padded_codes)
        approx_scores = self.module.scoring_function.aggregate_similarity(
            approx_similarity, query_scoring_mask=None, doc_scoring_mask=mask[:, None]
        )

        filtered_doc_idcs = []
        lengths = []
        iterator = zip(doc_idcs.packed_tensor.split(doc_idcs.lengths), approx_scores.split(doc_idcs.lengths))
        for doc_idcs, doc_scores in iterator:
            if doc_scores.shape[0] <= k:
                filtered_doc_idcs.append(doc_idcs)
            else:
                filtered_doc_idcs.append(doc_idcs[torch.topk(doc_scores, k, sorted=False)])
            lengths.append(filtered_doc_idcs[-1].shape[0])

        packed_filtered_doc_idcs = PackedTensor(torch.cat(filtered_doc_idcs), lengths)

        return packed_filtered_doc_idcs

    def _search(self, query_embeddings: BiEncoderEmbedding) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        query_embeddings = query_embeddings.to(self.device)
        centroid_scores, doc_idcs = self.candidate_retrieval(query_embeddings)
        # NOTE no idea why we do two filter steps (the first with a threshold, the second without)
        # filter step 1
        filtered_doc_idcs = self.filter_candidates(
            centroid_scores, doc_idcs, self.search_config.centroid_score_threshold, self.search_config.candidate_k
        )
        # filter step 2
        filtered_doc_idcs = self.filter_candidates(
            centroid_scores, filtered_doc_idcs, None, self.search_config.candidate_k // 4
        )

        # gather/decompress document embeddings
        doc_embedding_codes = self.packed_codes.lookup(filtered_doc_idcs.packed_tensor, 1)
        doc_embedding_residuals = self.packed_residuals.lookup(filtered_doc_idcs.packed_tensor, 1)
        doc_embeddings = self.residual_codec.decompress(doc_embedding_codes, doc_embedding_residuals)
        padded_doc_embeddings = doc_embeddings.to_padded_tensor()
        doc_scoring_mask = padded_doc_embeddings[..., 0] != 0

        # compute scores
        num_docs = filtered_doc_idcs.lengths
        doc_scores = self.module.scoring_function.forward(
            query_embeddings,
            BiEncoderEmbedding(padded_doc_embeddings, doc_scoring_mask, None),
            num_docs,
        )
        return doc_scores, filtered_doc_idcs.packed_tensor, num_docs


class PlaidSearchConfig(SearchConfig):

    search_class: Type[Searcher] = PlaidSearcher

    def __init__(
        self,
        k: int,
        candidate_k: int | None = None,
        n_cells: int | None = None,
        centroid_score_threshold: float | None = None,
    ) -> None:
        # https://github.com/stanford-futuredata/ColBERT/blob/7067ef598b5011edaa1f4a731a2c269dbac864e4/colbert/searcher.py#L106
        super().__init__(k)
        if candidate_k is None:
            if k <= 10:
                candidate_k = 256
            elif k <= 100:
                candidate_k = 1_024
            else:
                candidate_k = max(k * 4, 4_096)
        self.candidate_k = candidate_k
        if n_cells is None:
            if k <= 10:
                n_cells = 1
            elif k <= 100:
                n_cells = 2
            else:
                n_cells = 4
        self.n_cells = n_cells
        if centroid_score_threshold is None:
            if k <= 10:
                centroid_score_threshold = 0.5
            elif k <= 100:
                centroid_score_threshold = 0.45
            else:
                centroid_score_threshold = 0.4
        self.centroid_score_threshold = centroid_score_threshold
