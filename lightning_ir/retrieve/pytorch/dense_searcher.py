"""Torch-based Dense Searcher for Lightning IR Framework"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from ...modeling_utils.batching import _batch_pairwise_scoring
from ...models import ColConfig, DprConfig
from ..base.searcher import ExactSearchConfig, ExactSearcher
from .dense_indexer import TorchDenseIndexConfig

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderEmbedding, BiEncoderModule


class TorchDenseIndex:
    """Torch-based dense index for embeddings."""

    def __init__(self, index_dir: Path, similarity_function: Literal["dot", "cosine"], use_gpu: bool = False) -> None:
        """Initialize the TorchDenseIndex.

        Args:
            index_dir (Path): Directory where the index is stored.
            similarity_function (Literal["dot", "cosine"]): Similarity function to use for scoring.
            use_gpu (bool): Whether to use GPU for indexing. Defaults to False.
        Raises:
            ValueError: If the similarity function is not recognized.
        """
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
        """Score the embeddings against the index.

        Args:
            embeddings (torch.Tensor): The embeddings to score.
        Returns:
            torch.Tensor: The scores for the embeddings.
        """
        embeddings = embeddings.to(self.device)
        similarity = self.similarity_function(embeddings, self.index)
        return similarity

    @property
    def num_embeddings(self) -> int:
        """Get the number of embeddings in the index."""
        return self.index.shape[0]

    @staticmethod
    @_batch_pairwise_scoring
    @torch.autocast(device_type="cuda", enabled=False)
    def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the cosine similarity between two tensors.

        Args:
            x (torch.Tensor): First tensor.
            y (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: Cosine similarity scores.
        """
        return torch.nn.functional.cosine_similarity(x[:, None], y[None], dim=-1)

    @staticmethod
    @_batch_pairwise_scoring
    @torch.autocast(device_type="cuda", enabled=False)
    def dot_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the dot product similarity between two tensors.

        Args:
            x (torch.Tensor): First tensor.
            y (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: Dot product similarity scores.
        """
        return torch.matmul(x, y.T)

    def to_gpu(self) -> None:
        """Convert the index to GPU format."""
        self.index = self.index.to(self.device)


class TorchDenseSearcher(ExactSearcher):
    """Torch-based dense searcher for embeddings."""

    def __init__(
        self,
        index_dir: Path,
        search_config: TorchDenseSearchConfig,
        module: BiEncoderModule,
        use_gpu: bool = True,
    ) -> None:
        """Initialize the TorchDenseSearcher.

        Args:
            index_dir (Path): Directory where the index is stored.
            search_config (TorchDenseSearchConfig): Configuration for the dense search.
            module (BiEncoderModule): Bi-encoder module to use for searching.
            use_gpu (bool): Whether to use GPU for searching. Defaults to True.
        """
        self.search_config: TorchDenseSearchConfig
        self.index = TorchDenseIndex(index_dir, module.config.similarity_function, use_gpu)
        super().__init__(index_dir, search_config, module, use_gpu)
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    def to_gpu(self) -> None:
        """Move the searcher to the GPU if available."""
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
    """Configuration for the TorchDenseSearcher."""

    search_class = TorchDenseSearcher
    SUPPORTED_MODELS = {ColConfig.model_type, DprConfig.model_type}
