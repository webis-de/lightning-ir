from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import torch

if TYPE_CHECKING:
    from ..data import IndexBatch
    from ..bi_encoder import BiEncoderConfig, BiEncoderOutput


@dataclass
class IndexConfig:

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
            data["index_path"] = str(index_path)
            json.dump(data, f)


@dataclass
class FlatIndexConfig(IndexConfig):
    pass


@dataclass
class IVFPQIndexConfig(IndexConfig):
    num_train_embeddings: int
    num_centroids: int
    num_subquantizers: int = 16
    n_bits: int = 8


class Indexer(ABC):
    def __init__(
        self,
        index_path: Path,
        index_factory: str,
        index_config: IndexConfig,
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        self.index_path = index_path
        self.index_config = index_config
        self.bi_encoder_config = bi_encoder_config
        self.doc_ids = []
        self.doc_lengths = []
        self.num_embeddings = 0
        self.num_docs = 0
        self.verbose = verbose

        if self.bi_encoder_config.similarity_function == "l2":
            self.metric_type = faiss.METRIC_L2
        elif self.bi_encoder_config.similarity_function in ("cosine", "dot"):
            self.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(
                f"similarity_function {self.bi_encoder_config.similarity_function} "
                "unknown"
            )

        if self.bi_encoder_config.similarity_function == "cosine":
            index_factory = "L2norm," + index_factory
        self.index = faiss.index_factory(
            self.bi_encoder_config.embedding_dim, index_factory, self.metric_type
        )

        if torch.cuda.is_available():
            self.to_gpu()

        self.set_verbosity()

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        doc_embeddings = output.doc_embeddings
        if doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")
        doc_lengths = doc_embeddings.scoring_mask.sum(dim=1)
        embeddings = doc_embeddings.embeddings[doc_embeddings.scoring_mask]
        doc_ids = index_batch.doc_ids
        embeddings = self.process_embeddings(embeddings)

        if embeddings.shape[0]:
            self.index.add(embeddings.float().cpu())

        self.num_embeddings += embeddings.shape[0]
        self.num_docs += len(doc_ids)

        self.doc_lengths.append(doc_lengths.cpu())
        self.doc_ids.extend(doc_ids)

    def save(self) -> None:
        if self.num_embeddings != self.index.ntotal:
            raise ValueError("number of embeddings does not match index.ntotal")
        self.index_config.save(self.index_path)
        doc_lengths = torch.cat(self.doc_lengths)
        torch.save(doc_lengths, self.index_path / "doc_lengths.pt")
        (self.index_path / "doc_ids.txt").write_text("\n".join(self.doc_ids))
        if torch.cuda.is_available():
            self.index = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(self.index, str(self.index_path / "index.faiss"))

    @abstractmethod
    def to_gpu(self) -> None: ...

    @abstractmethod
    def to_cpu(self) -> None: ...

    @abstractmethod
    def set_verbosity(self) -> None: ...

    def process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings


class IVFPQIndexer(Indexer):
    def __init__(
        self,
        index_path: Path,
        index_config: IVFPQIndexConfig,
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        index_factory = (
            f"OPQ{index_config.num_subquantizers},"
            f"IVF{index_config.num_centroids}_HNSW32,"
            f"PQ{index_config.num_subquantizers}x{index_config.n_bits}"
        )
        super().__init__(
            index_path, index_factory, index_config, bi_encoder_config, verbose
        )
        self.index_config: IVFPQIndexConfig

        index_ivf_pq = faiss.downcast_index(self.index.index)
        index_ivf_pq.make_direct_map()

        self._train_embeddings = torch.full(
            (
                self.index_config.num_train_embeddings,
                self.bi_encoder_config.embedding_dim,
            ),
            torch.nan,
            dtype=torch.float32,
        )

    def to_gpu(self) -> None:
        clustering_index = faiss.index_cpu_to_all_gpus(
            faiss.IndexFlat(self.bi_encoder_config.embedding_dim, self.metric_type)
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

    def process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self._grab_train_embeddings(embeddings)
        self._train()
        return embeddings

    def _grab_train_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._train_embeddings is not None:
            # save training embeddings until num_train_embeddings is reached
            # if num_train_embeddings overflows, save the remaining embeddings
            start = self.num_embeddings
            end = start + embeddings.shape[0]
            if end > self.index_config.num_train_embeddings:
                end = self.index_config.num_train_embeddings
            length = end - start
            self._train_embeddings[start:end] = embeddings[:length]
            self.num_embeddings += length
            embeddings = embeddings[length:]
        return embeddings

    def _train(self, force: bool = False):
        if self._train_embeddings is not None and (
            force or self.num_embeddings >= self.index_config.num_train_embeddings
        ):
            if torch.isnan(self._train_embeddings).any():
                raise ValueError(
                    "corpus does not contain enough tokens/documents for training. "
                    "choose a larger corpus or reduce `num_train_embeddings`"
                )
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
        index_path: Path,
        index_config: FlatIndexConfig,
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        index_factory = "Flat"
        super().__init__(
            index_path, index_factory, index_config, bi_encoder_config, verbose
        )
        self.index_config: FlatIndexConfig

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def set_verbosity(self) -> None:
        self.index.verbose = self.verbose
