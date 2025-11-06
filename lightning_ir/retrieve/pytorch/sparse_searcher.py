"""Torch-based Sparse Searcher for Lightning IR Framework"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from ...modeling_utils.batching import _batch_pairwise_scoring
from ...models import SpladeConfig
from ..base.searcher import ExactSearchConfig, ExactSearcher
from .sparse_indexer import TorchSparseIndexConfig

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderEmbedding, BiEncoderModule


class TorchSparseIndex:
    """Torch-based sparse index for efficient retrieval."""

    def __init__(self, index_dir: Path, similarity_function: Literal["dot", "cosine"], use_gpu: bool = False) -> None:
        """Initialize the TorchSparseIndex.

        Args:
            index_dir (Path): Directory containing the index files.
            similarity_function (Literal["dot", "cosine"]): The similarity function to use.
            use_gpu (bool): Whether to use GPU for computations. Defaults to False.
        Raises:
            ValueError: If the similarity function is not recognized.
        """
        self.index = torch.load(index_dir / "index.pt", weights_only=True)
        self.config = TorchSparseIndexConfig.from_pretrained(index_dir)
        if similarity_function == "dot":
            self.similarity_function = self.dot_similarity
        elif similarity_function == "cosine":
            self.similarity_function = self.cosine_similarity
        else:
            raise ValueError("Unknown similarity function")
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute scores for the given embeddings.

        Args:
            embeddings (torch.Tensor): The embeddings to score.
        Returns:
            torch.Tensor: The computed scores.
        """
        embeddings = embeddings.to(self.index)
        similarity = self.similarity_function(embeddings, self.index).to_dense()
        return similarity

    @property
    def num_embeddings(self) -> int:
        """Get the number of embeddings in the index.

        Returns:
            int: The number of embeddings.
        """
        return self.index.shape[0]

    @staticmethod
    @_batch_pairwise_scoring
    @torch.autocast(device_type="cuda", enabled=False)
    def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two tensors.

        Args:
            x (torch.Tensor): The first tensor.
            y (torch.Tensor): The second tensor.
        Returns:
            torch.Tensor: The cosine similarity scores.
        """
        return y.matmul(x.T).T / (torch.norm(x, dim=-1)[:, None] * torch.norm(y, dim=-1)[None])

    @staticmethod
    @_batch_pairwise_scoring
    @torch.autocast(device_type="cuda", enabled=False)
    def dot_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dot product similarity between two tensors.

        Args:
            x (torch.Tensor): The first tensor.
            y (torch.Tensor): The second tensor.
        Returns:
            torch.Tensor: The dot product similarity scores.
        """
        return y.matmul(x.T).T

    def to_gpu(self) -> None:
        """Move the index to GPU if available."""
        self.index = self.index.to(self.device)


class TorchSparseSearcher(ExactSearcher):
    """Torch-based sparse searcher for Lightning IR framework."""

    def __init__(
        self,
        index_dir: Path,
        search_config: TorchSparseSearchConfig,
        module: BiEncoderModule,
        use_gpu: bool = True,
    ) -> None:
        """Initialize the TorchSparseSearcher.

        Args:
            index_dir (Path): Directory containing the index files.
            search_config (TorchSparseSearchConfig): Configuration for the searcher.
            module (BiEncoderModule): The BiEncoder module to use for scoring.
            use_gpu (bool): Whether to use GPU for computations. Defaults to True.
        """
        self.search_config: TorchSparseSearchConfig
        self.index = TorchSparseIndex(index_dir, module.config.similarity_function, use_gpu)
        super().__init__(index_dir, search_config, module, use_gpu)
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    def to_gpu(self) -> None:
        """Move the searcher and index to GPU if available."""
        super().to_gpu()
        self.index.to_gpu()

    def _score(self, query_embeddings: BiEncoderEmbedding) -> torch.Tensor:
        if query_embeddings.scoring_mask is None:
            embeddings = query_embeddings.embeddings[:, 0]
        else:
            embeddings = query_embeddings.embeddings[query_embeddings.scoring_mask]
        scores = self.index.score(embeddings)
        return scores


class TorchSparseSearchConfig(ExactSearchConfig):
    """Configuration for the Torch-based sparse searcher."""

    search_class = TorchSparseSearcher
    SUPPORTED_MODELS = {SpladeConfig.model_type}
