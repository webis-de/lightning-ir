from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Tuple, Type

import torch

from ..bi_encoder.model import BiEncoderEmbedding

if TYPE_CHECKING:
    from ..bi_encoder import BiEncoderModule, BiEncoderOutput


class Searcher(ABC):
    def __init__(
        self, index_dir: Path | str, search_config: SearchConfig, module: BiEncoderModule, use_gpu: bool = True
    ) -> None:
        super().__init__()
        self.index_dir = Path(index_dir)
        self.search_config = search_config
        self.module = module
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

        self.doc_ids = (self.index_dir / "doc_ids.txt").read_text().split()
        self.doc_lengths = torch.load(self.index_dir / "doc_lengths.pt")

        self.to_gpu()

        self.num_docs = len(self.doc_ids)
        self.cumulative_doc_lengths = torch.cumsum(self.doc_lengths, dim=0)

        if self.doc_lengths.shape[0] != self.num_docs or self.doc_lengths.sum() != self.num_embeddings:
            raise ValueError("doc_lengths do not match index")

    def to_gpu(self) -> None:
        self.doc_lengths = self.doc_lengths.to(self.device)

    @property
    @abstractmethod
    def num_embeddings(self) -> int: ...

    @abstractmethod
    def _search(self, query_embeddings: BiEncoderEmbedding) -> Tuple[torch.Tensor, torch.Tensor, List[int]]: ...

    def _filter_and_sort(
        self,
        doc_scores: torch.Tensor,
        doc_idcs: torch.Tensor | None,
        num_docs: Sequence[int] | None,
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        if (doc_idcs is None) != (num_docs is None):
            raise ValueError("doc_ids and num_docs must be both None or not None")
        if doc_idcs is None and num_docs is None:
            # assume we have searched the whole index
            k = min(self.search_config.k, doc_scores.shape[0])
            values, idcs = torch.topk(doc_scores.view(-1, self.num_docs), k)
            num_queries = values.shape[0]
            values = values.view(-1)
            idcs = idcs.view(-1)
            doc_ids = [self.doc_ids[doc_idx] for doc_idx in idcs.cpu()]
            return values, doc_ids, [k] * num_queries

        assert doc_idcs is not None and num_docs is not None
        per_query_doc_scores = torch.split(doc_scores, num_docs)
        per_query_doc_idcs = torch.split(doc_idcs, num_docs)
        new_num_docs = []
        _doc_scores = []
        doc_ids = []
        for query_idx, scores in enumerate(per_query_doc_scores):
            k = min(self.search_config.k, scores.shape[0])
            values, idcs = torch.topk(scores, k)
            _doc_scores.append(values)
            doc_ids.extend([self.doc_ids[doc_idx] for doc_idx in per_query_doc_idcs[query_idx][idcs].cpu()])
            new_num_docs.append(k)
        doc_scores = torch.cat(_doc_scores)
        return doc_scores, doc_ids, new_num_docs

    def search(self, output: BiEncoderOutput) -> Tuple[torch.Tensor, List[str], List[int]]:
        query_embeddings = output.query_embeddings
        if query_embeddings is None:
            raise ValueError("Expected query_embeddings in BiEncoderOutput")
        doc_scores, doc_idcs, num_docs = self._search(query_embeddings)
        doc_scores, doc_ids, num_docs = self._filter_and_sort(doc_scores, doc_idcs, num_docs)

        return doc_scores, doc_ids, num_docs


class SearchConfig:
    search_class: Type[Searcher] = Searcher

    def __init__(self, k: int = 10) -> None:
        self.k = k
