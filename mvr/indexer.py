import json
from pathlib import Path
from typing import NamedTuple, Sequence

import faiss
import torch

from .mvr import MVRConfig


class IndexConfig(NamedTuple):
    index_path: Path
    num_train_tokens: int
    num_centroids: int = 262144
    num_subquantizers: int = 16
    n_bits: int = 8

    @classmethod
    def from_pretrained(cls, index_path: Path) -> "IndexConfig":
        with open(index_path / "config.json", "r") as f:
            data = json.load(f)
            data["index_path"] = Path(data["index_path"])
            return cls(**data)

    def save(self, index_path: Path) -> None:
        index_path.mkdir(parents=True, exist_ok=True)
        with open(index_path / "config.json", "w") as f:
            data = self._asdict()
            data["index_path"] = str(data["index_path"])
            json.dump(data, f)


class Indexer:
    def __init__(
        self, index_config: IndexConfig, mvr_config: MVRConfig, verbose: bool = False
    ) -> None:
        self.index_config = index_config
        self.mvr_config = mvr_config
        self.doc_ids = []
        self.doc_lengths = []
        self.num_embeddings = 0
        self.num_docs = 0
        self.verbose = verbose

        if self.mvr_config.similarity_function == "l2":
            metric_type = faiss.METRIC_L2
        elif self.mvr_config.similarity_function in ("cosine", "dot"):
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(
                f"similarity_function {self.mvr_config.similarity_function} unknown"
            )
        index_factory = (
            f"OPQ{self.index_config.num_subquantizers},"
            f"IVF{self.index_config.num_centroids}_HNSW32,"
            f"PQ{self.index_config.num_subquantizers}x{self.index_config.n_bits}"
        )
        if self.mvr_config.similarity_function == "cosine":
            index_factory = "L2norm," + index_factory
        self.index = faiss.index_factory(
            self.mvr_config.embedding_dim, index_factory, metric_type
        )
        index_ivf_pq = faiss.downcast_index(self.index.index)
        index_ivf_pq.make_direct_map()

        for elem in (
            self.index,
            self.index.index,
            index_ivf_pq.cp,
            index_ivf_pq.pq,
            index_ivf_pq.quantizer,
        ):
            setattr(elem, "verbose", self.verbose)

        if torch.cuda.is_available():
            clustering_index = faiss.index_cpu_to_all_gpus(
                faiss.IndexFlat(self.mvr_config.embedding_dim, metric_type)
            )
            clustering_index.verbose = self.verbose
            index_ivf_pq.clustering_index = clustering_index

        self._train_embeddings = torch.empty(
            (self.index_config.num_train_tokens, self.mvr_config.embedding_dim),
            dtype=torch.float32,
        )

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
                # https://gist.github.com/mdouze/334ad6a979ac3637f6d95e9091356d3e
                # move index to cpu but leave quantizer on gpu
                self.index = faiss.index_gpu_to_cpu(self.index)
                index_ivf_pq = faiss.downcast_index(self.index.index)
                quantizer = index_ivf_pq.quantizer
                gpu_quantizer = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, quantizer
                )
                index_ivf_pq.quantizer = gpu_quantizer
            self.index.add(self._train_embeddings)
            self._train_embeddings = None
            self.index.verbose = False
            self.index.index.verbose = False

    def add(
        self,
        token_embeddings: torch.Tensor,
        doc_ids: Sequence[str],
        doc_lengths: torch.Tensor,
    ) -> None:
        token_embeddings = self._grab_train_embeddings(token_embeddings)
        self._train()

        if token_embeddings.shape[0]:
            self.index.add(token_embeddings.float().cpu())

        self.num_embeddings += token_embeddings.shape[0]
        self.num_docs += len(doc_ids)

        self.doc_lengths.append(doc_lengths.cpu())
        self.doc_ids.extend(doc_ids)

    def save(self) -> None:
        if self.num_embeddings != self.index.ntotal:
            raise ValueError("number of embeddings does not match index.ntotal")
        if not self.index.is_trained:
            self._train(force=True)
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
