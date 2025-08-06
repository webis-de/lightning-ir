"""
Lightning IR data module.

This module provides classes and configurations for indexing and searching using PLAID in the Lightning IR framework.

PLAID: `PLAID: An Efficient Engine for Late Interaction Retrieval
<https://dl.acm.org/doi/10.1145/3511808.3557325>`_
"""

from .plaid_indexer import PlaidIndexConfig, PlaidIndexer
from .plaid_searcher import PlaidSearchConfig, PlaidSearcher

__all__ = ["PlaidIndexConfig", "PlaidIndexer", "PlaidSearchConfig", "PlaidSearcher"]
