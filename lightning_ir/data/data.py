from typing import Any, Dict, NamedTuple, Sequence, Tuple

import torch
from transformers import BatchEncoding


class RunSample(NamedTuple):
    query_id: str
    query: str
    doc_ids: Tuple[str, ...]
    docs: Tuple[str, ...]
    targets: torch.Tensor | None = None
    qrels: Sequence[Dict[str, Any]] | None = None


class QuerySample(NamedTuple):
    query_id: str
    query: str

    @classmethod
    def from_ir_dataset_sample(cls, sample):
        return cls(sample[0], sample[1])


class DocSample(NamedTuple):
    doc_id: str
    doc: str

    @classmethod
    def from_ir_dataset_sample(cls, sample):
        return cls(sample[0], sample.default_text())


class BiEncoderRunBatch(NamedTuple):
    query_ids: Tuple[str, ...]
    query_encoding: BatchEncoding
    doc_ids: Tuple[Tuple[str, ...], ...]
    doc_encoding: BatchEncoding
    targets: torch.Tensor | None = None
    qrels: Dict[str, int] | None = None


class CrossEncoderRunBatch(NamedTuple):
    query_ids: Tuple[str, ...]
    doc_ids: Tuple[Tuple[str, ...], ...]
    encoding: BatchEncoding
    targets: torch.Tensor | None = None
    qrels: Dict[str, int] | None = None


class IndexBatch(NamedTuple):
    doc_ids: Tuple[str, ...]
    doc_encoding: BatchEncoding


class SearchBatch(NamedTuple):
    query_ids: Tuple[str, ...]
    query_encoding: BatchEncoding
