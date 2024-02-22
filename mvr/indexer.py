from pathlib import Path
from typing import NamedTuple
import json

import faiss
import numpy as np

from .mvr import MVRConfig


class IndexConfig(NamedTuple):
    index_path: Path
    num_train_tokens: int
    num_centroids: int = 65536
    num_subquantizers: int = 16
    code_size: int = 4

    @classmethod
    def from_pretrained(cls, index_path: Path) -> "IndexConfig":
        with open(index_path / "config.json", "r") as f:
            data = json.load(f)
            data["index_path"] = Path(data["index_path"])
            return cls(**data)

    def save_pretrained(self, index_path: Path) -> None:
        index_path.mkdir(parents=True, exist_ok=True)
        with open(index_path / "config.json", "w") as f:
            data = self._asdict()
            data["index_path"] = str(data["index_path"])
            json.dump(data, f)


class Indexer:
    def __init__(self, index_config: IndexConfig, mvr_config: MVRConfig) -> None:
        self.index_config = index_config
        self.mvr_config = mvr_config
        self.doc_ids = []
        self.doc_lengths = []
        self.num_embeddings = 0
        self.num_docs = 0

        if self.mvr_config.similarity_function == "l2":
            # coarse_quantizer = faiss.IndexFlatL2(self.mvr_config.embedding_dim)
            metric_type = faiss.METRIC_L2
        elif self.mvr_config.similarity_function in ("cosine", "dot"):
            # coarse_quantizer = faiss.IndexFlatIP(self.mvr_config.embedding_dim)
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(
                f"similarity_function {self.mvr_config.similarity_function} unknown"
            )
        index_factory = (
            f"IVF{self.index_config.num_centroids},"
            f"PQ{self.index_config.num_subquantizers}x{self.index_config.code_size}"
        )
        if self.mvr_config.similarity_function == "cosine":
            index_factory = "L2norm," + index_factory
        self.index = faiss.index_factory(
            self.mvr_config.embedding_dim, index_factory, metric_type
        )
        if self.mvr_config.similarity_function == "cosine":
            faiss.downcast_index(self.index.index).make_direct_map()
        else:
            self.index.make_direct_map()

        self._train_embeddings = np.empty(
            (self.index_config.num_train_tokens, self.mvr_config.embedding_dim),
            dtype=np.float32,
        )

    def _grab_train_embeddings(self, token_embeddings: np.ndarray) -> np.ndarray:
        if self._train_embeddings is not None:
            # save training embeddings until num_train_tokens is reached
            # if num_train_tokens overflows, save the remaining embeddings
            start = self.num_embeddings
            end = start + token_embeddings.shape[0]
            if end > self.index_config.num_train_tokens:
                end = self.index_config.num_train_tokens
            length = end - start
            self._train_embeddings[start:end] = token_embeddings[:length]
            self.num_embeddings += length
            token_embeddings = token_embeddings[length:]
        return token_embeddings

    def _train(self):
        if (
            self._train_embeddings is not None
            and self.num_embeddings >= self.index_config.num_train_tokens
        ):
            self.index.train(self._train_embeddings)
            self.index.add(self._train_embeddings)
            self._train_embeddings = None

    def add(
        self,
        token_embeddings: np.ndarray,
        doc_ids: np.ndarray,
        doc_lengths: np.ndarray,
    ) -> None:
        if doc_ids.dtype != np.uint8:
            raise ValueError("doc_ids must be of type np.uint8")
        self.doc_lengths.append(doc_lengths.astype(np.uint16))
        token_embeddings = self._grab_train_embeddings(token_embeddings)

        self._train()

        if token_embeddings.shape[0]:
            self.index.add(token_embeddings)

        self.num_embeddings += token_embeddings.shape[0]
        self.num_docs += doc_ids.shape[0]
        self.doc_ids.append(doc_ids)

    def save(self) -> None:
        self.index_config.index_path.mkdir(parents=True, exist_ok=True)
        self.index_config.save_pretrained(self.index_config.index_path)
        if not self.index.is_trained:
            raise ValueError("index is not trained")
        if self.num_embeddings != self.index.ntotal:
            raise ValueError("number of embeddings does not match index.ntotal")
        doc_ids_fp = np.memmap(
            self.index_config.index_path / "doc_ids.npy",
            dtype="uint8",
            mode="w+",
            shape=(self.num_docs, 20),
        )
        doc_lengths_fp = np.memmap(
            self.index_config.index_path / "doc_lengths.npy",
            dtype="uint16",
            mode="w+",
            shape=(self.num_docs,),
        )
        num_tokens = 0
        iterator = zip(self.doc_ids, self.doc_lengths or [None] * len(self.doc_ids))
        for doc_ids, doc_lengths in iterator:
            start = num_tokens
            end = start + doc_ids.shape[0]
            num_tokens += doc_ids.shape[0]
            doc_ids_fp[start:end] = doc_ids
            if doc_lengths is not None and doc_lengths_fp is not None:
                doc_lengths_fp[start:end] = doc_lengths
        doc_ids_fp.flush()
        if doc_lengths_fp is not None:
            doc_lengths_fp.flush()

        faiss.write_index(self.index, str(self.index_config.index_path / "index.faiss"))
