"""Torch-based Sparse Indexer for Lightning IR Framework"""

import array
from pathlib import Path

import torch

from ...bi_encoder import BiEncoderModule, BiEncoderOutput
from ...data import IndexBatch
from ...models import SpladeConfig
from ..base import IndexConfig, Indexer


class TorchSparseIndexer(Indexer):
    """Sparse indexer for bi-encoder models using PyTorch."""

    def __init__(
        self,
        index_dir: Path,
        index_config: "TorchSparseIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the TorchSparseIndexer.

        Args:
            index_dir (Path): Directory to store the index.
            index_config (TorchSparseIndexConfig): Configuration for the sparse index.
            module (BiEncoderModule): The bi-encoder module to use for indexing.
            verbose (bool): Whether to print verbose output. Defaults to False.
        """
        super().__init__(index_dir, index_config, module, verbose)
        self.crow_indices = array.array("L")
        self.crow_indices.append(0)
        self.col_indices = array.array("L")
        self.values = array.array("f")

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        """Add embeddings to the sparse index.

        Args:
            index_batch (IndexBatch): The batch containing the embeddings to index.
            output (BiEncoderOutput): The output from the bi-encoder model containing embeddings.
        Raises:
            ValueError: If doc_embeddings are not present in the output.
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
        num_docs = len(index_batch.doc_ids)
        self.doc_ids.extend(index_batch.doc_ids)

        crow_indices, col_indices, values = self.to_sparse_csr(embeddings)
        crow_indices = crow_indices[1:]  # remove the first element which is always 0
        crow_indices += self.crow_indices[-1]

        self.crow_indices.extend(crow_indices.cpu().tolist())
        self.col_indices.extend(col_indices.cpu().tolist())
        self.values.extend(values.cpu().tolist())

        self.doc_lengths.extend(doc_lengths.int().cpu().tolist())
        self.num_embeddings += embeddings.shape[0]
        self.num_docs += num_docs

    @staticmethod
    def to_sparse_csr(
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Convert embeddings to sparse CSR format.

        Args:
            embeddings (torch.Tensor): The embeddings tensor to convert.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Crow indices, column indices, and values of the sparse
                matrix.
        """
        token_idcs, dim_idcs = torch.nonzero(embeddings, as_tuple=True)
        crow_indices = (token_idcs + 1).bincount().cumsum(0)
        values = embeddings[token_idcs, dim_idcs]
        return crow_indices, dim_idcs, values

    def to_gpu(self) -> None:
        """Move the index to GPU if available."""
        pass

    def to_cpu(self) -> None:
        """Move the index to CPU."""
        pass

    def save(self) -> None:
        """Save the sparse index to disk."""
        super().save()
        index = torch.sparse_csr_tensor(
            torch.frombuffer(self.crow_indices, dtype=torch.int64),
            torch.frombuffer(self.col_indices, dtype=torch.int64),
            torch.frombuffer(self.values, dtype=torch.float32),
            torch.Size([self.num_embeddings, self.module.config.embedding_dim]),
        )
        torch.save(index, self.index_dir / "index.pt")


class TorchSparseIndexConfig(IndexConfig):
    """Configuration for the Torch-based sparse indexer."""

    indexer_class = TorchSparseIndexer
    SUPPORTED_MODELS = {SpladeConfig.model_type}
