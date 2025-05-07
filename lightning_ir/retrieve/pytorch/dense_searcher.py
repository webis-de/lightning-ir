from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from ...models import ColConfig, DprConfig
from ..base.searcher import ExactSearchConfig, ExactSearcher
from .dense_indexer import TorchDenseIndexConfig

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderEmbedding, BiEncoderModule


class TorchDenseIndex:
    def __init__(self, index_dir: Path, similarity_function: Literal["dot", "cosine"], use_gpu: bool = False) -> None:
        self.index = torch.load(index_dir / "index.pt", weights_only=True)
        self.config = TorchDenseIndexConfig.from_pretrained(index_dir)
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
        return torch.nn.functional.cosine_similarity(x[:, None], y[None], dim=-1)

    def dot_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.T)

    def to_gpu(self) -> None:
        self.index = self.index.to(self.device)


class TorchDenseSearcher(ExactSearcher):
    def __init__(
        self,
        index_dir: Path,
        search_config: TorchDenseSearchConfig,
        module: BiEncoderModule,
        use_gpu: bool = True,
    ) -> None:
        self.search_config: TorchDenseSearchConfig
        self.index = TorchDenseIndex(index_dir, module.config.similarity_function, use_gpu)
        super().__init__(index_dir, search_config, module, use_gpu)
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    def to_gpu(self) -> None:
        super().to_gpu()
        self.index.to_gpu()

    def _score(self, query_embeddings: BiEncoderEmbedding) -> torch.Tensor:
        if query_embeddings.scoring_mask is None:
            embeddings = query_embeddings.embeddings[:, 0]
        else:
            embeddings = query_embeddings.embeddings[query_embeddings.scoring_mask]
        scores = self.index.score(embeddings)
        return scores


class TorchDenseSearchConfig(ExactSearchConfig):
    search_class = TorchDenseSearcher
    SUPPORTED_MODELS = {ColConfig.model_type, DprConfig.model_type}
