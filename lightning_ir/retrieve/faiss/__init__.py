"""
Lightning IR data module.

This module provides classes and configurations for indexing and searching using FAISS in the Lightning IR framework.

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
