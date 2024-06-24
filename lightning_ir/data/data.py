from dataclasses import dataclass

from typing import Any, Dict, Sequence, Tuple

import torch


@dataclass
class RunSample:
    query_id: str
    query: str
    doc_ids: Tuple[str, ...]
    docs: Tuple[str, ...]
    targets: torch.Tensor | None = None
    qrels: Sequence[Dict[str, Any]] | None = None


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
    query_ids: Tuple[str, ...]
    queries: Tuple[str, ...]
    doc_ids: Tuple[Tuple[str, ...], ...]
    docs: Tuple[Tuple[str, ...], ...]


@dataclass
class TrainBatch(RankBatch):
    targets: torch.Tensor | None = None
    qrels: Dict[str, int] | None = None


@dataclass
class IndexBatch:
    doc_ids: Tuple[str, ...]
    docs: Tuple[Tuple[str, ...], ...]


@dataclass
class SearchBatch:
    query_ids: Tuple[str, ...]
    queries: Tuple[str, ...]
