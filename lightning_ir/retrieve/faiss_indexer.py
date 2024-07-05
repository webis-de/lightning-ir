import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Literal

import torch

from ..bi_encoder import BiEncoderConfig, BiEncoderOutput
from ..data import IndexBatch
from .indexer import IndexConfig, Indexer


class FaissFlatIndexConfig(IndexConfig):
    pass


class FaissIVFPQIndexConfig(IndexConfig):

    def __init__(
        self,
        similarity_function: None | Literal["cosine", "dot"] = None,
        num_train_embeddings: int | None = None,
        num_centroids: int = 262144,
        num_subquantizers: int = 16,
        n_bits: int = 8,
    ) -> None:
        super().__init__(similarity_function)
        self.num_train_embeddings = num_train_embeddings
        self.num_centroids = num_centroids
        self.num_subquantizers = num_subquantizers
        self.n_bits = n_bits


class FaissIndexer(Indexer):

    def __init__(
        self,
        index_dir: Path,
        index_config: IndexConfig,
        bi_encoder_config: BiEncoderConfig,
        index_factory: str,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, bi_encoder_config, verbose)
        import faiss

        self.index_factory = index_factory

        if self.index_config.similarity_function in ("cosine", "dot"):
            self.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(
                f"similarity_function {self.index_config.similarity_function} unknown"
            )

        if self.index_config.similarity_function == "cosine":
            index_factory = "L2norm," + index_factory
        self.index = faiss.index_factory(
            self.bi_encoder_config.embedding_dim, index_factory, self.metric_type
        )

        self.set_verbosity()

        if torch.cuda.is_available():
            self.to_gpu()

    @abstractmethod
    def to_gpu(self) -> None: ...

    @abstractmethod
    def to_cpu(self) -> None: ...

    @abstractmethod
    def set_verbosity(self) -> None: ...

    def process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings

    def save(self) -> None:
        super().save()
        import faiss

        if self.num_embeddings != self.index.ntotal:
            raise ValueError("number of embeddings does not match index.ntotal")
        if torch.cuda.is_available():
            self.index = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))

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

        self.doc_lengths.extend(doc_lengths.cpu().tolist())
        self.doc_ids.extend(doc_ids)


class FaissIVFPQIndexer(FaissIndexer):
    def __init__(
        self,
        index_dir: Path,
        index_config: FaissIVFPQIndexConfig,
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        import faiss

        index_factory = (
            f"OPQ{index_config.num_subquantizers},"
            f"IVF{index_config.num_centroids}_HNSW32,"
            f"PQ{index_config.num_subquantizers}x{index_config.n_bits}"
        )
        super().__init__(
            index_dir, index_config, bi_encoder_config, index_factory, verbose
        )
        self.index_config: FaissIVFPQIndexConfig

        index_ivf_pq = faiss.downcast_index(self.index.index)
        index_ivf_pq.make_direct_map()

        # default faiss values
        # https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/Clustering.h#L43
        max_points_per_centroid = 256
        self.num_train_embeddings = (
            index_config.num_train_embeddings
            or index_config.num_centroids * max_points_per_centroid
        )

        self._train_embeddings = torch.full(
            (
                self.num_train_embeddings,
                self.bi_encoder_config.embedding_dim,
            ),
            torch.nan,
            dtype=torch.float32,
        )

    def to_gpu(self) -> None:
        import faiss

        clustering_index = faiss.index_cpu_to_all_gpus(
            faiss.IndexFlat(self.bi_encoder_config.embedding_dim, self.metric_type)
        )
        clustering_index.verbose = self.verbose
        index_ivf_pq = faiss.downcast_index(self.index.index)
        index_ivf_pq.clustering_index = clustering_index

    def to_cpu(self) -> None:
        import faiss

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
        import faiss

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
            if end > self.num_train_embeddings:
                end = self.num_train_embeddings
            length = end - start
            self._train_embeddings[start:end] = embeddings[:length]
            self.num_embeddings += length
            embeddings = embeddings[length:]
        return embeddings

    def _train(self, force: bool = False):
        if self._train_embeddings is not None and (
            force or self.num_embeddings >= self.num_train_embeddings
        ):
            if torch.isnan(self._train_embeddings).any():
                warnings.warn(
                    "Corpus does not contain enough tokens/documents for training. "
                    "Removing NaN embeddings."
                )
                self._train_embeddings = self._train_embeddings[
                    ~torch.isnan(self._train_embeddings).any(dim=1)
                ]
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


class FaissFlatIndexer(FaissIndexer):
    def __init__(
        self,
        index_dir: Path,
        index_config: FaissFlatIndexConfig,
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        index_factory = "Flat"
        super().__init__(
            index_dir, index_config, bi_encoder_config, index_factory, verbose
        )
        self.index_config: FaissFlatIndexConfig

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def set_verbosity(self) -> None:
        self.index.verbose = self.verbose
