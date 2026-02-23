"""
Module for indexing and searching using Seismic.

This module provides the classes and configurations for indexing and searching using Seismic in the Lightning IR
framework.

Seismic accelerates search for learned sparse models by restructuring how the traditional inverted index stores and
processes data. Instead of evaluating every single document individually, it divides the inverted lists of terms into
manageable blocks and precomputes maximum score upper bounds for each of these blocks. During a search, the system
compares the query against these precalculated bounds to quickly evaluate entire groups of documents at once. If a
block's maximum possible score falls below the threshold needed to make it into the top results, Seismic safely skips
that entire block without calculating any individual document scores.

Seismic: `Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations
<https://dl.acm.org/doi/10.1145/3626772.3657769>`_
"""

from .seismic_indexer import SeismicIndexConfig, SeismicIndexer
from .seismic_searcher import SeismicSearchConfig, SeismicSearcher

__all__ = [
    "SeismicIndexConfig",
    "SeismicIndexer",
    "SeismicSearcher",
    "SeismicSearchConfig",
]
