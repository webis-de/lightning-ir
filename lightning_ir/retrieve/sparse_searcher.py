from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import torch

from lightning_ir.bi_encoder.model import BiEncoderEmbedding
from lightning_ir.bi_encoder.module import BiEncoderModule

from .searcher import SearchConfig, Searcher
from .sparse_indexer import SparseIndexConfig


class SparseIndex:

    def __init__(self, index_dir: Path) -> None:
        self.index = torch.load(index_dir / "index.pt")
        self.config = SparseIndexConfig.from_pretrained(index_dir)
        if self.config.similarity_function == "dot":
            self.similarity_function = self.dot_similarity
        elif self.config.similarity_function == "cosine":
            self.similarity_function = self.cosine_similarity
        else:
            raise ValueError("Unknown similarity function")

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        sparse_embeddings = embeddings.to_sparse_coo()
        similarity = self.similarity_function(sparse_embeddings, self.index).to_dense()
        return similarity

    @property
    def num_embeddings(self) -> int:
        return self.index.shape[0]

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot_product = self.dot_similarity(x, y)
        dot_product = dot_product / (torch.norm(x, dim=-1) * torch.norm(y, dim=-1))
        return -1 * torch.cdist(x, y).squeeze(-2)

    def dot_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.T)


@dataclass
class SparseSearchConfig(SearchConfig):
    pass


class SparseSearcher(Searcher):

    def __init__(
        self,
        index_dir: Path,
        search_config: SparseSearchConfig,
        module: BiEncoderModule,
    ) -> None:
        self.search_config: SparseSearchConfig
        self.index = SparseIndex(index_dir)
        super().__init__(index_dir, search_config, module)
        self.doc_token_idcs = (
            torch.arange(self.doc_lengths.shape[0])
            .to(self.doc_lengths)
            .repeat_interleave(self.doc_lengths)
        )

    @property
    def num_embeddings(self) -> int:
        return self.index.num_embeddings

    def _search(
        self, query_embeddings: BiEncoderEmbedding
    ) -> Tuple[torch.Tensor, None, None]:
        embeddings = query_embeddings.embeddings[query_embeddings.scoring_mask]
        query_lengths = query_embeddings.scoring_mask.sum(-1)
        scores = self.index.score(embeddings)

        # aggregate doc token scores
        scores = torch.scatter_reduce(
            torch.zeros(scores.shape[0], self.num_docs, device=scores.device),
            1,
            self.doc_token_idcs[None].expand_as(scores),
            scores,
            "amax",
        )

        query_token_idcs = (
            torch.arange(query_lengths.shape[0])
            .to(query_lengths)
            .repeat_interleave(query_lengths)
        )
        # aggregate query token scores
        scores = torch.scatter_reduce(
            torch.zeros(query_lengths.shape[0], self.num_docs, device=scores.device),
            0,
            query_token_idcs[:, None].expand_as(scores),
            scores,
            self.module.config.query_aggregation_function,
        ).view(-1)
        return scores, None, None
