import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import torch

from ..bi_encoder.model import BiEncoderConfig


@dataclass
class IndexConfig:
    index_path: Path

    @classmethod
    def from_pretrained(cls, index_path: Path) -> "IndexConfig":
        with open(index_path / "config.json", "r") as f:
            data = json.load(f)
            data["index_path"] = Path(data["index_path"])
            return cls(**data)

    def save(self, index_path: Path) -> None:
        index_path.mkdir(parents=True, exist_ok=True)
        with open(index_path / "config.json", "w") as f:
            data = self.__dict__.copy()
            data["index_path"] = str(data["index_path"])
            json.dump(data, f)


@dataclass
class FlatIndexConfig(IndexConfig):
    @classmethod
    def from_pretrained(cls, index_path: Path) -> "FlatIndexConfig":
        with open(index_path / "config.json", "r") as f:
            data = json.load(f)
            data["index_path"] = Path(data["index_path"])
            return cls(**data)


@dataclass
class IVFPQIndexConfig(IndexConfig):
    num_train_tokens: int
    num_centroids: int = 262144
    num_subquantizers: int = 16
    n_bits: int = 8

    @classmethod
    def from_pretrained(cls, index_path: Path) -> "IVFPQIndexConfig":
        with open(index_path / "config.json", "r") as f:
            data = json.load(f)
            data["index_path"] = Path(data["index_path"])
            return cls(**data)


class Indexer(ABC):
    def __init__(
        self,
        index_factory: str,
        index_config: IndexConfig,
        mvr_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        self.index_config = index_config
        self.mvr_config = mvr_config
        self.doc_ids = []
        self.doc_lengths = []
        self.num_embeddings = 0
        self.num_docs = 0
        self.verbose = verbose

        if self.mvr_config.similarity_function == "l2":
            self.metric_type = faiss.METRIC_L2
        elif self.mvr_config.similarity_function in ("cosine", "dot"):
            self.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(
                f"similarity_function {self.mvr_config.similarity_function} unknown"
            )

        if self.mvr_config.similarity_function == "cosine":
            index_factory = "L2norm," + index_factory
        self.index = faiss.index_factory(
            self.mvr_config.embedding_dim, index_factory, self.metric_type
        )

        if torch.cuda.is_available():
            self.to_gpu()

        self.set_verbosity()

    def add(
        self,
        token_embeddings: torch.Tensor,
        doc_ids: Sequence[str],
        doc_lengths: torch.Tensor,
    ) -> None:
        token_embeddings = self.process_token_embeddings(token_embeddings)

        if token_embeddings.shape[0]:
            self.index.add(token_embeddings.float().cpu())

        self.num_embeddings += token_embeddings.shape[0]
        self.num_docs += len(doc_ids)

        self.doc_lengths.append(doc_lengths.cpu())
        self.doc_ids.extend(doc_ids)

    def save(self) -> None:
        if self.num_embeddings != self.index.ntotal:
            raise ValueError("number of embeddings does not match index.ntotal")
        self.index_config.index_path.mkdir(parents=True, exist_ok=True)
        self.index_config.save(self.index_config.index_path)
        doc_lengths = torch.cat(self.doc_lengths)
        torch.save(doc_lengths, self.index_config.index_path / "doc_lengths.pt")
        (self.index_config.index_path / "doc_ids.txt").write_text(
            "\n".join(self.doc_ids)
        )
        if torch.cuda.is_available():
            self.index = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(self.index, str(self.index_config.index_path / "index.faiss"))

    @abstractmethod
    def to_gpu(self) -> None: ...

    @abstractmethod
    def to_cpu(self) -> None: ...

    @abstractmethod
    def set_verbosity(self) -> None: ...

    def process_token_embeddings(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return token_embeddings


class IVFPQIndexer(Indexer):
    def __init__(
        self,
        index_config: IVFPQIndexConfig,
        mvr_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        index_factory = (
            f"OPQ{index_config.num_subquantizers},"
            f"IVF{index_config.num_centroids}_HNSW32,"
            f"PQ{index_config.num_subquantizers}x{index_config.n_bits}"
        )
        super().__init__(index_factory, index_config, mvr_config, verbose)
        self.index_config: IVFPQIndexConfig

        index_ivf_pq = faiss.downcast_index(self.index.index)
        index_ivf_pq.make_direct_map()

        self._train_embeddings = torch.empty(
            (self.index_config.num_train_tokens, self.mvr_config.embedding_dim),
            dtype=torch.float32,
        )

    def to_gpu(self) -> None:
        clustering_index = faiss.index_cpu_to_all_gpus(
            faiss.IndexFlat(self.mvr_config.embedding_dim, self.metric_type)
        )
        clustering_index.verbose = self.verbose
        index_ivf_pq = faiss.downcast_index(self.index.index)
        index_ivf_pq.clustering_index = clustering_index

    def to_cpu(self) -> None:
        # https://gist.github.com/mdouze/334ad6a979ac3637f6d95e9091356d3e
        # move index to cpu but leave quantizer on gpu
        self.index = faiss.index_gpu_to_cpu(self.index)
        index_ivf_pq = faiss.downcast_index(self.index.index)
        quantizer = index_ivf_pq.quantizer
        gpu_quantizer = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(), 0, quantizer
        )
        index_ivf_pq.quantizer = gpu_quantizer

    def set_verbosity(self) -> None:
        index_ivf_pq = faiss.downcast_index(self.index.index)
        for elem in (
            self.index,
            self.index.index,
            index_ivf_pq.cp,
            index_ivf_pq.pq,
            index_ivf_pq.quantizer,
        ):
            setattr(elem, "verbose", self.verbose)

    def process_token_embeddings(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        token_embeddings = self._grab_train_embeddings(token_embeddings)
        self._train()
        return token_embeddings

    def _grab_train_embeddings(self, token_embeddings: torch.Tensor) -> torch.Tensor:
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

    def _train(self, force: bool = False):
        if self._train_embeddings is not None and (
            force or self.num_embeddings >= self.index_config.num_train_tokens
        ):
            self.index.train(self._train_embeddings)
            if torch.cuda.is_available():
                self.to_cpu()
            self.index.add(self._train_embeddings)
            self._train_embeddings = None
            self.index.verbose = False
            self.index.index.verbose = False

    def save(self) -> None:
        if not self.index.is_trained:
            self._train(force=True)
        return super().save()


class FlatIndexer(Indexer):
    def __init__(
        self,
        index_config: FlatIndexConfig,
        mvr_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        index_factory = "Flat"
        super().__init__(index_factory, index_config, mvr_config, verbose)
        self.index_config: FlatIndexConfig

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def set_verbosity(self) -> None:
        self.index.verbose = self.verbose
