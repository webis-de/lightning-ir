from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch


@dataclass
class RankSample:
    query_id: str
    query: str
    doc_ids: Sequence[str]
    docs: Sequence[str]
    targets: torch.Tensor | None = None
    qrels: List[Dict[str, Any]] | None = None


@dataclass
class QuerySample:
    query_id: str
    query: str

    @classmethod
    def from_ir_dataset_sample(cls, sample):
        return cls(sample[0], sample[1])


@dataclass
class DocSample:
    doc_id: str
    doc: str

    @classmethod
    def from_ir_dataset_sample(cls, sample):
        return cls(sample[0], sample.default_text())


@dataclass
class RankBatch:
    queries: Sequence[str]
    docs: Sequence[Sequence[str]]
    query_ids: Sequence[str] | None = None
    doc_ids: Sequence[Sequence[str]] | None = None
    qrels: List[Dict[str, int]] | None = None


@dataclass
class TrainBatch(RankBatch):
    targets: torch.Tensor | None = None


@dataclass
class IndexBatch:
    doc_ids: Sequence[str]
    docs: Sequence[str]


@dataclass
class SearchBatch:
    query_ids: Sequence[str]
    queries: Sequence[str]
    doc_ids: Sequence[Sequence[str]] | None = None
    qrels: List[Dict[str, int]] | None = None
