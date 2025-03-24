"""
Basic sample classes for Lightning IR.

This module defines the basic samples classes for Lightning IR. A sample is single entry in a dataset and can be grouped
into batches for processing.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from ir_datasets.formats.base import GenericDoc, GenericQuery


@dataclass
class RankSample:
    """A sample of ranking data containing a query, a ranked list of documents, and optionally targets and qrels.

    :param query_id: Id of the query
    :type query_id: str
    :param query: Query text
    :type query_id: str
    :param doc_ids: List of document ids
    :type doc_ids: Sequence[str]
    :param docs: List of document texts
    :type docs: Sequence[str]
    :param targets: Optional list of target labels denoting the relevane of a document for the query
    :type targets: torch.Tensor, optional
    :param qrels: Optional list of dictionaries mapping document ids to relevance labels
    """

    query_id: str
    query: str
    doc_ids: Sequence[str]
    docs: Sequence[str]
    targets: torch.Tensor | None = None
    qrels: List[Dict[str, Any]] | None = None


@dataclass
class QuerySample:
    """A sample of query data containing a query and its id.

    :param query_id: Id of the query
    :type query_id: str
    :param query: Query text
    :type query_id: str
    """

    query_id: str
    query: str

    @classmethod
    def from_ir_dataset_sample(cls, sample: GenericQuery) -> "QuerySample":
        """Create a QuerySample from a an ir_datasets sample.

        :param sample: ir_datasets sample
        :type sample: GenericQuery
        :return: Query sample
        :rtype: QuerySample
        """
        return cls(sample[0], sample[1])


@dataclass
class DocSample:
    """A sample of document data containing a document and its id.

    :param doc_id: Id of the document
    :type doc_id: str
    :param doc: Document text
    :type doc: str
    """

    doc_id: str
    doc: str

    @classmethod
    def from_ir_dataset_sample(cls, sample: GenericDoc, text_fields: Sequence[str] | None = None) -> "DocSample":
        """Create a DocSample from an ir_datasets sample.

        :param sample: ir_datasets sample
        :type sample: GenericDoc
        :param text_fields: Optional fields to parse the text. If None uses the samples ``default_text()``
            defaults to None
        :type text_fields: Sequence[str] | None, optional
        :return: Doc sample
        :rtype: DocSample
        """
        if text_fields is not None:
            return cls(sample[0], " ".join(getattr(sample, field) for field in text_fields))
        return cls(sample[0], sample.default_text())


@dataclass
class RankBatch:
    """A batch of ranking data combining multiple :py:class:`.RankSample` instances

    :param queries: List of query texts
    :type queries: Sequence[str]
    :param docs: List of list of document texts
    :type docs: Sequence[Sequence[str]]
    :param query_ids: Optional list of query ids
    :type query_ids: Sequence[str], optional
    :param doc_ids: Optional list of list of document ids
    :type doc_ids: Sequence[Sequence[str]], optional
    :param qrels: Optional list of dictionaries mapping document ids to relevance labels
    :type qrels: List[Dict[str, Any]], optional
    """

    queries: Sequence[str]
    docs: Sequence[Sequence[str]]
    query_ids: Sequence[str] | None = None
    doc_ids: Sequence[Sequence[str]] | None = None
    qrels: List[Dict[str, int]] | None = None


@dataclass
class TrainBatch(RankBatch):
    """A batch of ranking data that combines multiple :py:class:`.RankSample` instances

    :param queries: List of query texts
    :type queries: Sequence[str]
    :param docs: List of list of document texts
    :type docs: Sequence[Sequence[str]]
    :param query_ids: Optional list of query ids
    :type query_ids: Sequence[str], optional
    :param doc_ids: Optional list of list of document ids
    :type doc_ids: Sequence[Sequence[str]], optional
    :param qrels: Optional list of dictionaries mapping document ids to relevance labels
    :type qrels: List[Dict[str, Any]], optional
    :param targets: Optional list of target labels denoting the relevane of a document for the query
    :type targets: torch.Tensor, optional
    """

    targets: torch.Tensor | None = None


@dataclass
class IndexBatch:
    """A batch of index that combines multiple :py:class:`.DocSample` instances

    :param doc_ids: List of document ids
    :type doc_ids: Sequence[str]
    :param docs: List of document texts
    :type docs: Sequence[str]
    """

    doc_ids: Sequence[str]
    docs: Sequence[str]


@dataclass
class SearchBatch:
    """A batch of search data that combines multiple :py:class:`.QuerySample` instances. Optionaly includes document ids
    and qrels.

    :param query_ids: List of query ids
    :type query_ids: Sequence[str]
    :param queries: List of query texts
    :type queries: Sequence[str]
    :param doc_ids: Optional list of list of document ids
    :type doc_ids: Sequence[Sequence[str]], optional
    :param qrels: Optional list of dictionaries mapping document ids to relevance labels
    :type qrels: List[Dict[str, Any]], optional
    """

    query_ids: Sequence[str]
    queries: Sequence[str]
    doc_ids: Sequence[Sequence[str]] | None = None
    qrels: List[Dict[str, int]] | None = None
