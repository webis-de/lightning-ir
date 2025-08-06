from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import torch

from ...bi_encoder.bi_encoder_model import BiEncoderEmbedding
from ...models import ColConfig
from ..base.packed_tensor import PackedTensor
from ..base.searcher import SearchConfig, Searcher
from .plaid_indexer import PlaidIndexConfig
from .residual_codec import ResidualCodec

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderModule, BiEncoderOutput


class PlaidSearcher(Searcher):
    def __init__(
        self, index_dir: Path | str, search_config: PlaidSearchConfig, module: BiEncoderModule, use_gpu: bool = False
    ) -> None:
        super().__init__(index_dir, search_config, module, use_gpu)
        self.residual_codec = ResidualCodec.from_pretrained(
            PlaidIndexConfig.from_pretrained(self.index_dir), self.index_dir, device=self.device
        )

        self.codes = torch.load(self.index_dir / "codes.pt", weights_only=True).to(self.device)
        self.residuals = (
            torch.load(self.index_dir / "residuals.pt", weights_only=True).view(self.codes.shape[0], -1).to(self.device)
        )
        self.packed_codes = PackedTensor(self.codes, lengths=self.doc_lengths.tolist()).to(self.device)
        self.packed_residuals = PackedTensor(self.residuals, lengths=self.doc_lengths.tolist()).to(self.device)

        # code_idx to embedding_idcs mapping
        sorted_codes, embedding_idcs = self.codes.sort()
        num_embeddings_per_code = torch.bincount(sorted_codes, minlength=self.residual_codec.num_centroids).tolist()

        # code_idx to doc_idcs mapping
        embedding_idx_to_doc_idx = torch.arange(self.num_docs, device=self.device).repeat_interleave(self.doc_lengths)
        full_doc_ivf = embedding_idx_to_doc_idx[embedding_idcs]
        doc_ivf_lengths = []
        unique_doc_idcs = []
        for doc_idcs in full_doc_ivf.split(num_embeddings_per_code):
            unique_doc_idcs.append(doc_idcs.unique())
            doc_ivf_lengths.append(unique_doc_idcs[-1].shape[0])
        self.code_to_doc_ivf = PackedTensor(torch.cat(unique_doc_idcs), lengths=doc_ivf_lengths)

        # doc_idx to code_idcs mapping
        sorted_doc_idcs, doc_idx_to_code_idx = torch.sort(self.code_to_doc_ivf)
        code_idcs = torch.arange(self.residual_codec.num_centroids, device=self.device).repeat_interleave(
            torch.tensor(self.code_to_doc_ivf.lengths, device=self.device)
        )[doc_idx_to_code_idx]
        num_codes_per_doc = torch.bincount(sorted_doc_idcs, minlength=self.num_docs)
        self.doc_to_code_ivf = PackedTensor(code_idcs, lengths=num_codes_per_doc.tolist())

        self.search_config: PlaidSearchConfig

    def _centroid_candidate_retrieval(self, query_embeddings: BiEncoderEmbedding) -> Tuple[PackedTensor, PackedTensor]:
        # grab top `n_cells` neighbor cells for all query embeddings
        # `num_queries x query_length x num_centroids`
        centroid_scores = (
            query_embeddings.embeddings.to(self.residual_codec.centroids)
            @ self.residual_codec.centroids.transpose(0, 1)[None]
        ).to(self.device)
        query_scoring_mask = query_embeddings.scoring_mask
        centroid_scores = centroid_scores.masked_fill(~query_scoring_mask[..., None], 0)
        _, codes = torch.topk(centroid_scores, self.search_config.n_cells, dim=-1, sorted=False)
        packed_codes = codes[query_embeddings.scoring_mask].view(-1)
        code_lengths = (query_embeddings.scoring_mask.sum(-1) * self.search_config.n_cells).tolist()

        # grab document idcs for all cells
        packed_doc_idcs = self.code_to_doc_ivf.lookup(packed_codes, code_lengths, unique=True)

        # NOTE no idea why we do two filter steps (the first with a threshold, the second without)
        # filter step 1
        _, filtered_doc_idcs = self._filter_candidates(
            centroid_scores=centroid_scores,
            doc_idcs=packed_doc_idcs,
            threshold=self.search_config.centroid_score_threshold,
            k=self.search_config.candidate_k,
            query_scoring_mask=query_scoring_mask,
        )
        # filter step 2
        filtered_scores, filtered_doc_idcs = self._filter_candidates(
            centroid_scores=centroid_scores,
            doc_idcs=filtered_doc_idcs,
            threshold=None,
            k=self.search_config.candidate_k // 4,
            query_scoring_mask=query_scoring_mask,
        )
        return filtered_scores, filtered_doc_idcs

    def _filter_candidates(
        self,
        centroid_scores: torch.Tensor,
        doc_idcs: PackedTensor,
        threshold: float | None,
        k: int,
        query_scoring_mask: torch.Tensor,
    ) -> Tuple[PackedTensor, PackedTensor]:
        num_query_vecs = centroid_scores.shape[1]
        num_centroids = centroid_scores.shape[-1]

        # repeat query centroid scores for each document
        # `num_docs x num_query_vecs x num_centroids + 1`
        # NOTE we pad values such that the codes with -1 padding index 0 values
        expanded_centroid_scores = torch.nn.functional.pad(
            centroid_scores.repeat_interleave(torch.tensor(doc_idcs.lengths, device=self.device), dim=0), (0, 1)
        )

        # grab codes for each document
        code_idcs = self.doc_to_code_ivf.lookup(doc_idcs, 1)
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
        approx_scores = self.module.model.aggregate_similarity(
            approx_similarity,
            query_scoring_mask=query_scoring_mask,
            doc_scoring_mask=mask[:, None],
            num_docs=doc_idcs.lengths,
        )
        packed_approx_scores = PackedTensor(approx_scores, lengths=doc_idcs.lengths)
        filtered_scores, filtered_doc_idcs = self._filter_and_sort(packed_approx_scores, doc_idcs, k)
        return filtered_scores, filtered_doc_idcs

    def _reconstruct_doc_embeddings(self, candidate_doc_idcs: PackedTensor) -> BiEncoderEmbedding:
        doc_embedding_codes = self.packed_codes.lookup(candidate_doc_idcs, 1)
        doc_embedding_residuals = self.packed_residuals.lookup(candidate_doc_idcs, 1)
        doc_embeddings = self.residual_codec.decompress(doc_embedding_codes, doc_embedding_residuals)
        padded_doc_embeddings = doc_embeddings.to_padded_tensor()
        doc_scoring_mask = padded_doc_embeddings[..., 0] != 0
        return BiEncoderEmbedding(padded_doc_embeddings, doc_scoring_mask, None)

    def search(self, output: BiEncoderOutput) -> Tuple[PackedTensor, List[List[str]]]:
        query_embeddings = output.query_embeddings
        if query_embeddings is None:
            raise ValueError("Expected query_embeddings in BiEncoderOutput")
        query_embeddings = query_embeddings.to(self.device)

        _, candidate_idcs = self._centroid_candidate_retrieval(query_embeddings)
        num_docs = candidate_idcs.lengths

        # compute scores
        doc_embeddings = self._reconstruct_doc_embeddings(candidate_idcs)
        scores = self.module.model.score(query_embeddings, doc_embeddings, num_docs)

        scores, doc_idcs = self._filter_and_sort(PackedTensor(scores, lengths=candidate_idcs.lengths), candidate_idcs)
        doc_ids = [
            [self.doc_ids[doc_idx] for doc_idx in _doc_ids.tolist()] for _doc_ids in doc_idcs.split(doc_idcs.lengths)
        ]
        return scores, doc_ids


class PlaidSearchConfig(SearchConfig):

    search_class = PlaidSearcher
    SUPPORTED_MODELS = {ColConfig.model_type}

    def __init__(
        self,
        k: int,
        candidate_k: int = 256,
        n_cells: int = 1,
        centroid_score_threshold: float = 0.5,
    ) -> None:
        super().__init__(k)
        self.candidate_k = candidate_k
        self.n_cells = n_cells
        self.centroid_score_threshold = centroid_score_threshold
