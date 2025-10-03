"""Plaid Indexer using fast-plaid library for Lightning IR Framework"""

import warnings
from pathlib import Path

import torch
from fast_plaid import search

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
        self._train_embeddings = None
        self._num_buffered = 0

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        """Add embeddings from the index batch to the Plaid index.

        Args:
            index_batch (IndexBatch): Batch of data containing embeddings to be indexed.
            output (BiEncoderOutput): Output from the BiEncoder module containing embeddings.
        Raises:
            ValueError: If the output does not contain document embeddings.
        """
        doc_embeddings = output.doc_embeddings
        if output.doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")
        # doc_embeddings = output.doc_embeddings.embeddings.detach()

        num_train = self.index_config.num_train_embeddings

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

        if self.index is None:
            if self._train_embeddings is None:
                self._train_embeddings = embeddings.new_full(
                    (num_train, doc_embeddings.embeddings.shape[1], doc_embeddings.embeddings.shape[2]), float("nan")
                )
                self._num_buffered = 0

            n = doc_embeddings.embeddings.shape[0]
            start = self._num_buffered
            end = min(num_train, start + n)
            length = end - start
            self._train_embeddings[start:end] = doc_embeddings.embeddings[:length]
            self._num_buffered += length

            if self._num_buffered < num_train:
                return

            train_embs = self._train_embeddings
            if torch.isnan(train_embs).any():
                train_embs = train_embs[~torch.isnan(train_embs).any(dim=1)]

            config_path = self.index_dir / "config.json"
            config_bytes = config_path.read_bytes() if config_path.exists() else None
            self.index = search.FastPlaid(index=str(self.index_dir))
            self.index.create(
                documents_embeddings=train_embs.detach(),
                kmeans_niters=self.index_config.k_means_iters,
                nbits=self.index_config.n_bits,
                n_samples_kmeans=num_train,
                seed=self.index_config.seed,
            )
            if config_bytes is not None and not config_path.exists():
                config_path.write_bytes(config_bytes)

            if length < n:
                self.index.update(documents_embeddings=doc_embeddings.embeddings[length:].detach())
            self._train_embeddings = None
        else:
            self.index.update(documents_embeddings=doc_embeddings.embeddings.detach())

    def finalize(self):
        """Finalize index creation with buffered embeddings if not enough were provided."""
        if self.index is not None:
            return
        if self._train_embeddings is None:
            return

        num_train = self.index_config.num_train_embeddings
        if self._num_buffered < num_train:
            warnings.warn(
                f"Not enough doc_embeddings provided for Plaid index creation: "
                f"expected {num_train}, got {self._num_buffered}. Index will be created with fewer embeddings."
            )
        train_embs = self._train_embeddings

        config_path = self.index_dir / "config.json"
        config_bytes = config_path.read_bytes() if config_path.exists() else None
        self.index = search.FastPlaid(index=str(self.index_dir))

        self.index.create(
            documents_embeddings=train_embs.detach().cpu(),
            kmeans_niters=self.index_config.k_means_iters,
            nbits=self.index_config.n_bits,
            n_samples_kmeans=num_train,
            seed=self.index_config.seed,
        )

        if config_bytes is not None and not config_path.exists():
            config_path.write_bytes(config_bytes)
        self._train_embeddings = None

    def save(self) -> None:
        """Save the index configuration and document IDs to the index directory."""
        super().save()
        if self.index is None:
            self.finalize()


class PlaidIndexConfig(IndexConfig):
    """Configuration class for Plaid indexers in the Lightning IR framework."""

    indexer_class = PlaidIndexer
    SUPPORTED_MODELS = {ColConfig.model_type}

    def __init__(
        self,
        num_centroids: int,
        num_train_embeddings: int | None = None,
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
        max_points_per_centroid = 256
        self.num_centroids = num_centroids
        self.num_train_embeddings = num_train_embeddings or num_centroids * max_points_per_centroid
        self.k_means_iters = k_means_iters
        self.n_bits = n_bits
        self.seed = seed
