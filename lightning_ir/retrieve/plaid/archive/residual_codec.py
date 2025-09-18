"""Residual Codec for Plaid Indexing and Retrieval"""

from __future__ import annotations

import pathlib
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load

from ...base.packed_tensor import PackedTensor

if TYPE_CHECKING:
    from .plaid_indexer import PlaidIndexConfig


class ResidualCodec:
    """Residual Codec for Plaid, a residual-based search method for efficient retrieval."""

    def __init__(
        self,
        index_config: PlaidIndexConfig,
        centroids: torch.Tensor,
        bucket_cutoffs: torch.Tensor,
        bucket_weights: torch.Tensor,
        verbose: bool = False,
    ) -> None:
        """Initialize the ResidualCodec.

        Args:
            index_config (PlaidIndexConfig): Configuration for the Plaid indexer.
            centroids (torch.Tensor): The centroids used for indexing.
            bucket_cutoffs (torch.Tensor): The cutoffs for the residual buckets.
            bucket_weights (torch.Tensor): The weights for the residual buckets.
            verbose (bool): Whether to print verbose output. Defaults to False.
        """
        self.index_config = index_config
        self.verbose = verbose

        self.centroids = centroids
        self.bucket_cutoffs = bucket_cutoffs
        self.bucket_weights = bucket_weights

        self.arange_bits = torch.arange(0, self.index_config.n_bits, dtype=torch.uint8, device=self.centroids.device)
        self.reversed_bit_map = self._compute_reverse_bit_map()
        keys_per_byte = 8 // self.index_config.n_bits
        self.decompression_lookup_table = torch.tensor(
            list(product(list(range(len(self.bucket_weights))), repeat=keys_per_byte)),
            device=self.centroids.device,
            dtype=torch.uint8,
        )

        self.residual_dim = max(1, centroids.shape[-1] // 8 * index_config.n_bits)

        self._packbits_cpp = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, num_centroids={self.num_centroids})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def dim(self) -> int:
        """Get the dimensionality of the centroids."""
        return self.centroids.shape[-1]

    @property
    def num_centroids(self) -> int:
        """Get the number of centroids."""
        return self.centroids.shape[0]

    @classmethod
    def train(
        cls, index_config: PlaidIndexConfig, train_embeddings: torch.Tensor, verbose: bool = False
    ) -> "ResidualCodec":
        """Train the ResidualCodec using the provided training embeddings.

        Args:
            index_config (PlaidIndexConfig): Configuration for the Plaid indexer.
            train_embeddings (torch.Tensor): The embeddings to use for training the codec.
            verbose (bool): Whether to print verbose output. Defaults to False.
        Returns:
            ResidualCodec: An instance of the ResidualCodec trained on the provided embeddings.
        """
        train_embeddings = train_embeddings[torch.randperm(train_embeddings.shape[0])]
        num_hold_out_embeddings = int(min(0.05 * train_embeddings.shape[0], 2**15))
        train_embeddings, holdout_embeddings = train_embeddings.split(
            [train_embeddings.shape[0] - num_hold_out_embeddings, num_hold_out_embeddings]
        )

        centroids = cls._train_kmeans(train_embeddings, index_config, verbose)
        bucket_cutoffs, bucket_weights = cls._compute_buckets(centroids, holdout_embeddings, index_config)

        return cls(index_config, centroids, bucket_cutoffs, bucket_weights, verbose)

    @staticmethod
    def _train_kmeans(embeddings: torch.Tensor, index_config: PlaidIndexConfig, verbose: bool = False) -> torch.Tensor:
        import faiss

        kmeans = faiss.Kmeans(
            embeddings.shape[-1],
            index_config.num_centroids,
            niter=index_config.k_means_iters,
            gpu=torch.cuda.is_available(),
            verbose=verbose,
            seed=index_config.seed,
        )
        # TODO why normalize?
        kmeans.train(embeddings.numpy())
        return torch.nn.functional.normalize(torch.from_numpy(kmeans.centroids), dim=-1)

    def _packbits(self, residuals: torch.Tensor) -> torch.Tensor:
        if residuals.device == torch.device("cuda"):
            raise NotImplementedError("CUDA not supported for packbits")
        residuals_packed = torch.from_numpy(np.packbits(np.asarray(residuals.contiguous().flatten())))
        return residuals_packed

    @staticmethod
    def _compute_buckets(
        centroids: torch.Tensor, holdout_embeddings: torch.Tensor, index_config: PlaidIndexConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        holdout_embeddings_codes = ResidualCodec._compress_into_codes(centroids, holdout_embeddings)
        holdout_embeddings_centroids = centroids[holdout_embeddings_codes]

        holdout_residual = holdout_embeddings - holdout_embeddings_centroids
        avg_residual = holdout_residual.abs().mean(dim=0)

        num_options = 2**index_config.n_bits
        quantiles = torch.arange(0, num_options, device=avg_residual.device) * (1 / num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)

        bucket_cutoffs = holdout_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = holdout_residual.float().quantile(bucket_weights_quantiles)
        return bucket_cutoffs, bucket_weights

    def _compute_reverse_bit_map(self) -> torch.Tensor:
        # We reverse the residual bits because arange_bits as
        # currently constructed produces results with the reverse
        # of the expected endianness

        reversed_bit_map = []
        mask = (1 << self.index_config.n_bits) - 1
        for i in range(256):
            # The reversed byte
            z = 0
            for j in range(8, 0, -self.index_config.n_bits):
                # Extract a subsequence of length n bits
                x = (i >> (j - self.index_config.n_bits)) & mask

                # Reverse the endianness of each bit subsequence (e.g. 10 -> 01)
                y = 0
                for k in range(self.index_config.n_bits - 1, -1, -1):
                    y += ((x >> (self.index_config.n_bits - k - 1)) & 1) * (2**k)

                # Set the corresponding bits in the output byte
                z |= y
                if j > self.index_config.n_bits:
                    z <<= self.index_config.n_bits
            reversed_bit_map.append(z)
        return torch.tensor(reversed_bit_map, dtype=torch.uint8, device=self.centroids.device)

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        """Load the necessary C++ extensions for the ResidualCodec.

        Args:
            cls: The class to load the extensions for.
            use_gpu (bool): Whether to use GPU for the extensions.
        """
        if hasattr(cls, "loaded_extensions") or not use_gpu:
            return

        decompress_residuals_cpp = load(
            name="decompress_residuals_cpp",
            sources=[
                str(pathlib.Path(__file__).parent.resolve() / "csrc" / "decompress_residuals.cpp"),
                str(pathlib.Path(__file__).parent.resolve() / "csrc" / "decompress_residuals.cu"),
            ],
        )
        cls.decompress_residuals = decompress_residuals_cpp.decompress_residuals_cpp

        cls.loaded_extensions = True

    @classmethod
    def from_pretrained(
        cls, index_config: PlaidIndexConfig, index_dir: Path, device: torch.device | None = None
    ) -> "ResidualCodec":
        """Load a ResidualCodec from the specified directory.

        Args:
            index_config (PlaidIndexConfig): Configuration for the Plaid indexer.
            index_dir (Path): Directory containing the saved codec files.
            device (torch.device | None): Device to load the codec onto. Defaults to None, which uses the CPU.
        Returns:
            ResidualCodec: An instance of the ResidualCodec loaded from the specified directory.
        """
        centroids_path = index_dir / "centroids.pt"
        buckets_path = index_dir / "buckets.pt"

        centroids = torch.load(
            centroids_path, map_location=str(device) if device is not None else "cpu", weights_only=True
        )
        bucket_cutoffs, bucket_weights = torch.load(
            buckets_path, map_location=str(device) if device is not None else "cpu", weights_only=True
        )

        return cls(
            index_config=index_config,
            centroids=centroids,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights,
        )

    def save(self, index_dir: Path):
        """Save the ResidualCodec to the specified directory.

        Args:
            index_dir (Path): Directory to save the codec files.
        Raises:
            ValueError: If residual_codec is None.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        centroids_path = index_dir / "centroids.pt"
        buckets_path = index_dir / "buckets.pt"

        torch.save(self.centroids.half(), centroids_path)
        torch.save((self.bucket_cutoffs, self.bucket_weights), buckets_path)

    @staticmethod
    def _compress_into_codes(centroids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        codes = []
        batch_size = 2**29 // centroids.shape[0]
        for batch in embeddings.split(batch_size):
            indices = (centroids @ batch.transpose(-1, -2)).argmax(dim=0)
            codes.append(indices)
        return torch.cat(codes)

    def compress_into_codes(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compress embeddings into codes using the centroids.

        Args:
            embeddings (torch.Tensor): The embeddings to compress.
        Returns:
            torch.Tensor: The compressed codes.
        """
        return self._compress_into_codes(self.centroids, embeddings)

    def compress(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress embeddings into codes and residuals.

        Args:
            embeddings (torch.Tensor): The embeddings to compress.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the compressed codes and residuals.
        """
        embeddings = embeddings.to(self.centroids.device)
        codes = self.compress_into_codes(embeddings)
        centroids = self.centroids[codes]
        residuals = self.binarize(embeddings - centroids)
        return codes, residuals

    def binarize(self, residuals: torch.Tensor) -> torch.Tensor:
        """Binarize the residuals using the bucket cutoffs and weights.

        Args:
            residuals (torch.Tensor): The residuals to binarize.
        Returns:
            torch.Tensor: The binarized residuals.
        """
        buckets = torch.bucketize(residuals.float(), self.bucket_cutoffs).to(dtype=torch.uint8)
        buckets_expanded = buckets.unsqueeze(-1).expand(*buckets.size(), self.index_config.n_bits)
        bucket_bits = buckets_expanded >> self.arange_bits  # divide by 2^bit for each bit position
        bucket_binary = bucket_bits & 1  # apply mod 2 to binarize

        residuals_packed = self._packbits(bucket_binary)
        residuals_packed = residuals_packed.reshape(residuals.size(0), max(1, self.dim // 8 * self.index_config.n_bits))

        return residuals_packed

    def decompress(self, codes: PackedTensor, compressed_residuals: PackedTensor) -> PackedTensor:
        """Decompress the codes and residuals into embeddings.

        Args:
            codes (PackedTensor): The packed tensor containing the codes.
            compressed_residuals (PackedTensor): The packed tensor containing the compressed residuals.
        Returns:
            PackedTensor: The decompressed embeddings.
        """
        centroids = self.centroids[codes]
        residuals = self.reversed_bit_map[compressed_residuals.long().view(-1)].view_as(compressed_residuals)
        residuals = self.decompression_lookup_table[residuals.long()]
        residuals = residuals.view(residuals.shape[0], -1)
        residuals = self.bucket_weights[residuals.long()]
        embeddings = centroids + residuals
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return PackedTensor(embeddings, lengths=codes.lengths)
