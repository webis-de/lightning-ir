"""
Module for indexing and searching using FAISS.

This module provides classes and configurations for indexing and searching using FAISS in the Lightning IR framework.

FAISS (Facebook AI Similarity Search) is an open-source library that provides the foundational infrastructure for
making dense vector retrieval models like DPR fast and scalable in production. Rather than creating the vector
embeddings itself, FAISS acts as a highly optimized database engine designed exclusively for storing and searching
through massive collections of dense vectors. It uses approximate nearest neighbor algorithms, such as Product
Quantization and Hierarchical Navigable Small World graphs, to drastically reduce the time and memory required to find
the closest matching vectors to a search query. By intelligently grouping and compressing these mathematical
representations, FAISS allows search systems to bypass exhaustive comparisons and deliver near-instant results across
billions of documents.

FAISS: `The Faiss library
<https://arxiv.org/abs/2401.08281>`_
"""

from .faiss_indexer import (
    FaissFlatIndexConfig,
    FaissFlatIndexer,
    FaissIVFIndexConfig,
    FaissIVFIndexer,
    FaissIVFPQIndexConfig,
    FaissIVFPQIndexer,
    FaissPQIndexConfig,
    FaissPQIndexer,
)
from .faiss_searcher import FaissSearchConfig, FaissSearcher

__all__ = [
    "FaissFlatIndexConfig",
    "FaissFlatIndexer",
    "FaissIVFIndexConfig",
    "FaissIVFIndexer",
    "FaissIVFPQIndexConfig",
    "FaissIVFPQIndexer",
    "FaissPQIndexConfig",
    "FaissPQIndexer",
    "FaissSearchConfig",
    "FaissSearcher",
]
