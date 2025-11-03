"""Plaid Indexer using fast-plaid library for Lightning IR Framework"""

from pathlib import Path

import torch

from ...bi_encoder import BiEncoderModule, BiEncoderOutput
from ...data import IndexBatch
from ...models import ColConfig
from ..base import IndexConfig, Indexer


class PlaidIndexer(Indexer):
    """Indexer for Plaid using fast-plaid library."""

    def __init__(
        self,
        index_dir: Path,
        index_config: "PlaidIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the PlaidIndexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (PlaidIndexConfig): Configuration for the Plaid indexer.
            module (BiEncoderModule): The BiEncoder module used for indexing.
            verbose (bool): Whether to print verbose output during indexing. Defaults to False.
        """
        super().__init__(index_dir, index_config, module, verbose)
        self.index_config: PlaidIndexConfig
        self.index = None
        self.embeddings = []

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        """Add embeddings from the index batch to the Plaid index.

        Args:
            index_batch (IndexBatch): Batch of data containing embeddings to be indexed.
            output (BiEncoderOutput): Output from the BiEncoder module containing embeddings.
        Raises:
            ValueError: If the output does not contain document embeddings.
        """
        doc_embeddings = output.doc_embeddings
        if doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")

        if doc_embeddings.scoring_mask is None:
            doc_lengths = torch.ones(
                doc_embeddings.embeddings.shape[0], device=doc_embeddings.device, dtype=torch.int32
            )
            embeddings = doc_embeddings.embeddings[:, 0]
        else:
            doc_lengths = doc_embeddings.scoring_mask.sum(dim=1)
            embeddings = doc_embeddings.embeddings[doc_embeddings.scoring_mask]
        doc_ids = index_batch.doc_ids

        self.num_embeddings += embeddings.shape[0]
        self.num_docs += len(doc_ids)

        self.doc_lengths.extend(doc_lengths.int().cpu().tolist())
        self.doc_ids.extend(doc_ids)

        self.embeddings.extend(embeddings.cpu().split(doc_lengths.int().cpu().tolist()))

    def save(self) -> None:
        """Save the index configuration and document IDs to the index directory."""
        from fast_plaid import search

        index = search.FastPlaid(index=str(self.index_dir))

        index.create(
            documents_embeddings=self.embeddings,
            kmeans_niters=self.index_config.k_means_iters,
            nbits=self.index_config.n_bits,
            seed=self.index_config.seed,
        )

        super().save()


class PlaidIndexConfig(IndexConfig):
    """Configuration class for Plaid indexers in the Lightning IR framework."""

    indexer_class = PlaidIndexer
    SUPPORTED_MODELS = {ColConfig.model_type}

    def __init__(
        self,
        num_centroids: int,
        k_means_iters: int = 4,
        n_bits: int = 2,
        seed: int = 42,
    ) -> None:
        """Initialize the PlaidIndexConfig.

        Args:
            num_centroids (int): Number of centroids for the Plaid index.
            num_train_embeddings (int | None): Number of embeddings to use for training the index. If None, it will
                be set later. Defaults to None.
            k_means_iters (int): Number of iterations for k-means clustering. Defaults to 4.
            n_bits (int): Number of bits for the residual codec. Defaults to 2.
            seed (int): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__()
        self.num_centroids = num_centroids
        self.k_means_iters = k_means_iters
        self.n_bits = n_bits
        self.seed = seed
