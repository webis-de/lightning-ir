"""
Base module for indexing and searching.

This module provides the base classes and configurations for indexing and searching in the Lightning IR framework.
"""

from .indexer import IndexConfig, Indexer
from .searcher import SearchConfig, Searcher

__all__ = [
    "IndexConfig",
    "Indexer",
    "SearchConfig",
    "Searcher",
]
