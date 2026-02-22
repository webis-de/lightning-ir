"""
Module for dense and sparseindexing and searching using PyTorch.

The PyTorch Indexing and Searching unifies both dense and sparse retrieval pipelines into a single streamlined
interface. It utilizes PyTorch tensors and GPU acceleration to efficiently encode, store, and query documents, offering
dedicated architectures for both continuous vector embeddings and high-dimensional sparse representations.

This module provides the classes and configurations for dense and sparse indexing and searching using PyTorch in the
Lightning IR framework.
"""

from .dense_indexer import TorchDenseIndexConfig, TorchDenseIndexer
from .dense_searcher import TorchDenseSearchConfig, TorchDenseSearcher
from .sparse_indexer import TorchSparseIndexConfig, TorchSparseIndexer
from .sparse_searcher import TorchSparseSearchConfig, TorchSparseSearcher

__all__ = [
    "TorchDenseIndexConfig",
    "TorchDenseIndexer",
    "TorchDenseSearchConfig",
    "TorchDenseSearcher",
    "TorchSparseIndexConfig",
    "TorchSparseIndexer",
    "TorchSparseSearchConfig",
    "TorchSparseSearcher",
]
