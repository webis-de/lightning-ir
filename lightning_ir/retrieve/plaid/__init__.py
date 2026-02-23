"""
Module for indexing and searching using PLAID.

This module provides classes and configurations for indexing and searching using PLAID in the Lightning IR framework.

PLAID (Performance-optimized Late Interaction Driver) is a optimized search engine designed to drastically
accelerate multi-vector models like ColBERT without sacrificing their state-of-the-art accuracy. While late interaction
architectures provide precision by comparing every query word to every document word, doing this
exhaustively across massive databases is computationally slow. PLAID solves this by grouping document token vectors
into a smaller set of representative "centroids" during the indexing phase. When a search occurs, the system first
performs a rapid filter by matching the query against these lightweight centroid IDs, pruning away irrelevant
documents. It then only executes the expensive, exact token-level mathematical comparisons on the small handful of
highly relevant candidate passages that remain.

PLAID: `PLAID: An Efficient Engine for Late Interaction Retrieval
<https://dl.acm.org/doi/10.1145/3511808.3557325>`_
"""

from .plaid_indexer import PlaidIndexConfig, PlaidIndexer
from .plaid_searcher import PlaidSearchConfig, PlaidSearcher

__all__ = ["PlaidIndexConfig", "PlaidIndexer", "PlaidSearchConfig", "PlaidSearcher"]
