from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Tuple

import torch

from .searcher import SearchConfig, Searcher
from .sparse_indexer import SparseIndexConfig

if TYPE_CHECKING:
    from ..bi_encoder import BiEncoderEmbedding, BiEncoderModule


class SparseIndex:
    def __init__(self, index_dir: Path, similarity_function: Literal["dot", "cosine"], use_gpu: bool = False) -> None:
        self.index = torch.load(index_dir / "index.pt")
        self.config = SparseIndexConfig.from_pretrained(index_dir)
        if similarity_function == "dot":
            self.similarity_function = self.dot_similarity
        elif similarity_function == "cosine":
            self.similarity_function = self.cosine_similarity
        else:
            raise ValueError("Unknown similarity function")
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.to(self.device)
        similarity = self.similarity_function(embeddings, self.index).to_dense()
        return similarity

    @property
    def num_embeddings(self) -> int:
        return self.index.shape[0]

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot_product = self.dot_similarity(x, y)
        dot_product = dot_product / (torch.norm(x, dim=-1) * torch.norm(y, dim=-1))
        return -1 * torch.cdist(x, y).squeeze(-2)

    def dot_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(y, x.T).T

    def to_gpu(self) -> None:
        self.index = self.index.to(self.device)


class SparseSearcher(Searcher):
    def __init__(
        self,
        index_dir: Path,
        search_config: SparseSearchConfig,
        module: BiEncoderModule,
        use_gpu: bool = True,
    ) -> None:
        self.search_config: SparseSearchConfig
        self.index = SparseIndex(index_dir, module.config.similarity_function, use_gpu)
        super().__init__(index_dir, search_config, module, use_gpu)
        self.doc_token_idcs = (
            torch.arange(self.doc_lengths.shape[0]).to(self.doc_lengths).repeat_interleave(self.doc_lengths)
        )
        self.use_gpu = use_gpu
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    @property
    def doc_is_single_vector(self) -> bool:
        return self.cumulative_doc_lengths[-1].item() == self.cumulative_doc_lengths.shape[0]

    def to_gpu(self) -> None:
        super().to_gpu()
        self.index.to_gpu()

    @property
    def num_embeddings(self) -> int:
        return self.index.num_embeddings

    def _search(self, query_embeddings: BiEncoderEmbedding) -> Tuple[torch.Tensor, None, None]:
        embeddings = query_embeddings.embeddings[query_embeddings.scoring_mask]
        query_lengths = query_embeddings.scoring_mask.sum(-1)
        scores = self.index.score(embeddings)

        # aggregate doc token scores
        if not self.doc_is_single_vector:
            scores = torch.scatter_reduce(
                torch.zeros(scores.shape[0], self.num_docs, device=scores.device),
                1,
                self.doc_token_idcs[None].expand_as(scores),
                scores,
                "amax",
            )

        # aggregate query token scores
        query_is_single_vector = (query_lengths == 1).all()
        if not query_is_single_vector:
            query_token_idcs = torch.arange(query_lengths.shape[0]).to(query_lengths).repeat_interleave(query_lengths)
            scores = torch.scatter_reduce(
                torch.zeros(query_lengths.shape[0], self.num_docs, device=scores.device),
                0,
                query_token_idcs[:, None].expand_as(scores),
                scores,
                self.module.config.query_aggregation_function,
            )
        scores = scores.reshape(-1)
        return scores, None, None


class SparseSearchConfig(SearchConfig):
    search_class = SparseSearcher
