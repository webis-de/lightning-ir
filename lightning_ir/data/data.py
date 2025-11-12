"""
Basic sample classes for Lightning IR.

This module defines the basic samples classes for Lightning IR. A sample is single entry in a dataset and can be grouped
into batches for processing.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from ir_datasets.formats.base import GenericDoc, GenericQuery


@dataclass
class RankSample:
    """A sample of ranking data containing a query, a ranked list of documents, and optionally targets and qrels.

    Attributes:
        query_id (str): Id of the query.
        query (str): Query text.
        doc_ids (Sequence[str]): list of document ids.
        docs (Sequence[str]): list of document texts.
        targets (torch.Tensor): Optional list of target labels denoting the relevance of a document for the query.
            Defaults to None.
        qrels (list[dict[str, Any]]): Optional list of dictionaries mapping document ids to relevance labels.
            Defaults to None.
    """

    query_id: str
    query: str
    doc_ids: Sequence[str]
    docs: Sequence[str]
    targets: torch.Tensor | None = None
    qrels: list[dict[str, Any]] | None = None


@dataclass
class QuerySample:
    """A sample of query data containing a query and its id.

    Attributes:
        query_id (str): Id of the query.
        query (str): Query text.
        qrels (list[dict[str, Any]] | None): Optional list of dictionaries mapping document ids to relevance labels.
            Defaults to None.
    """

    query_id: str
    query: str
    qrels: list[dict[str, Any]] | None = None

    @classmethod
    def from_ir_dataset_sample(cls, sample: GenericQuery) -> "QuerySample":
        """Create a QuerySample from a an ir_datasets sample.

        Args:
            sample (GenericQuery): ir_datasets sample.
        Returns:
            QuerySample: Query sample.
        """
        return cls(str(sample[0]), sample.default_text())


@dataclass
class DocSample:
    """A sample of document data containing a document and its id.

    Attributes:
        doc_id (str): Id of the document.
        doc (str): Document text.
    """

    doc_id: str
    doc: str

    @classmethod
    def from_ir_dataset_sample(cls, sample: GenericDoc, text_fields: Sequence[str] | None = None) -> "DocSample":
        """Create a DocSample from an ir_datasets sample.

        Args:
            sample (GenericDoc): ir_datasets sample.
            text_fields (Sequence[str] | None): Optional fields to parse the text. If None uses the sample's
                `default_text()`. Defaults to None.
        Returns:
            DocSample: Document sample.
        """
        if text_fields is not None:
            return cls(sample[0], " ".join(getattr(sample, field) for field in text_fields))
        return cls(str(sample[0]), sample.default_text())


@dataclass
class RankBatch:
    """A batch of ranking data combining multiple :py:class:`.RankSample` instances

    Attributes:
        queries (Sequence[str]): list of query texts.
        docs (Sequence[Sequence[str]]): list of list of document texts.
        query_ids (Sequence[str] | None): Optional list of query ids. Defaults to None.
        doc_ids (Sequence[Sequence[str]] | None): Optional list of list of document ids. Defaults to None.
        qrels (list[dict[str, int]] | None): Optional list of dictionaries mapping document ids to relevance labels.
            Defaults to None.
    """

    queries: Sequence[str]
    docs: Sequence[Sequence[str]]
    query_ids: Sequence[str] | None = None
    doc_ids: Sequence[Sequence[str]] | None = None
    qrels: list[dict[str, int]] | None = None


@dataclass
class TrainBatch(RankBatch):
    """A batch of ranking data that combines multiple :py:class:`.RankSample` instances

    Attributes:
        queries (Sequence[str]): list of query texts.
        docs (Sequence[Sequence[str]]): list of list of document texts.
        query_ids (Sequence[str] | None): Optional list of query ids. Defaults to None.
        doc_ids (Sequence[Sequence[str]] | None): Optional list of list of document ids. Defaults to None.
        qrels (list[dict[str, int]] | None): Optional list of dictionaries mapping document ids to relevance labels.
            Defaults to None.
        targets (torch.Tensor | None): Optional list of target labels denoting the relevance of a document for the
            query. Defaults to None.
    """

    targets: torch.Tensor | None = None


@dataclass
class IndexBatch:
    """A batch of index that combines multiple :py:class:`.DocSample` instances

    Attributes:
        doc_ids (Sequence[str]): list of document ids.
        docs (Sequence[str]): list of document texts.
    """

    doc_ids: Sequence[str]
    docs: Sequence[str]


@dataclass
class SearchBatch:
    """A batch of search data that combines multiple :py:class:`.QuerySample` instances. Optionaly includes document ids
    and qrels.

    Attributes:
        query_ids (Sequence[str]): list of query ids.
        queries (Sequence[str]): list of query texts.
        doc_ids (Sequence[Sequence[str]] | None): Optional list of list of document ids. Defaults to None.
        qrels (list[dict[str, int]] | None): Optional list of dictionaries mapping document ids to relevance labels.
            Defaults to None.
    """

    query_ids: Sequence[str]
    queries: Sequence[str]
    doc_ids: Sequence[Sequence[str]] | None = None
    qrels: list[dict[str, int]] | None = None
