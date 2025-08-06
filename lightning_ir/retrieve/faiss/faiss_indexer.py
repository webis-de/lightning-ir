import warnings
from pathlib import Path
from typing import Type

import torch

from ...bi_encoder import BiEncoderModule, BiEncoderOutput
from ...data import IndexBatch
from ...models import ColConfig, DprConfig
from ..base import IndexConfig, Indexer


class FaissIndexer(Indexer):
    INDEX_FACTORY: str

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, module, verbose)
        import faiss

        similarity_function = self.module.config.similarity_function
        if similarity_function in ("cosine", "dot"):
            self.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"similarity_function {similarity_function} unknown")

        index_factory = self.INDEX_FACTORY.format(**index_config.to_dict())
        if similarity_function == "cosine":
            index_factory = "L2norm," + index_factory
        self.index = faiss.index_factory(self.module.config.embedding_dim, index_factory, self.metric_type)

        self.set_verbosity()

        if torch.cuda.is_available():
            self.to_gpu()

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def set_verbosity(self, verbose: bool | None = None) -> None:
        self.index.verbose = self.verbose if verbose is None else verbose

    def process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings

    def save(self) -> None:
        super().save()
        import faiss

        if self.num_embeddings != self.index.ntotal:
            raise ValueError("number of embeddings does not match index.ntotal")
        if torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu"):
            self.index = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        doc_embeddings = output.doc_embeddings
        if doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")
        if doc_embeddings.scoring_mask is None:
            doc_lengths = torch.ones(
                doc_embeddings.embeddings.shape[0], device=doc_embeddings.device, dtype=torch.int32
            )
            embeddings = doc_embeddings.embeddings[:, 0]
        else:
            doc_lengths = doc_embeddings.scoring_mask.sum(dim=1)
            embeddings = doc_embeddings.embeddings[doc_embeddings.scoring_mask]
        doc_ids = index_batch.doc_ids
        embeddings = self.process_embeddings(embeddings)

        if embeddings.shape[0]:
            self.index.add(embeddings.float().cpu())

        self.num_embeddings += embeddings.shape[0]
        self.num_docs += len(doc_ids)

        self.doc_lengths.extend(doc_lengths.int().cpu().tolist())
        self.doc_ids.extend(doc_ids)


class FaissFlatIndexer(FaissIndexer):
    INDEX_FACTORY = "Flat"

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissFlatIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, module, verbose)
        self.index_config: FaissFlatIndexConfig

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass


class _FaissTrainIndexer(FaissIndexer):

    INDEX_FACTORY = ""  # class only acts as mixin

    def __init__(
        self,
        index_dir: Path,
        index_config: "_FaissTrainIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, module, verbose)
        if index_config.num_train_embeddings is None:
            raise ValueError("num_train_embeddings must be set")
        self.num_train_embeddings = index_config.num_train_embeddings

        self._train_embeddings: torch.Tensor | None = torch.full(
            (self.num_train_embeddings, self.module.config.embedding_dim), torch.nan, dtype=torch.float32
        )

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
            end = min(self.num_train_embeddings, start + embeddings.shape[0])
            length = end - start
            self._train_embeddings[start:end] = embeddings[:length]
            self.num_embeddings += length
            embeddings = embeddings[length:]
        return embeddings

    def _train(self, force: bool = False):
        if self._train_embeddings is None:
            return
        if not force and self.num_embeddings < self.num_train_embeddings:
            return
        if torch.isnan(self._train_embeddings).any():
            warnings.warn("Corpus contains less tokens/documents than num_train_embeddings. Removing NaN embeddings.")
            self._train_embeddings = self._train_embeddings[~torch.isnan(self._train_embeddings).any(dim=1)]
        self.index.train(self._train_embeddings)
        if torch.cuda.is_available():
            self.to_cpu()
        self.index.add(self._train_embeddings)
        self._train_embeddings = None
        self.set_verbosity(False)

    def save(self) -> None:
        if not self.index.is_trained:
            self._train(force=True)
        return super().save()


class FaissIVFIndexer(_FaissTrainIndexer):
    INDEX_FACTORY = "IVF{num_centroids},Flat"

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissIVFIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        # default faiss values
        # https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/Clustering.h#L43
        max_points_per_centroid = 256
        index_config.num_train_embeddings = (
            index_config.num_train_embeddings or index_config.num_centroids * max_points_per_centroid
        )
        super().__init__(index_dir, index_config, module, verbose)

        import faiss

        ivf_index = faiss.extract_index_ivf(self.index)
        if hasattr(ivf_index, "quantizer"):
            quantizer = ivf_index.quantizer
            if hasattr(faiss.downcast_index(quantizer), "hnsw"):
                downcasted_quantizer = faiss.downcast_index(quantizer)
                downcasted_quantizer.hnsw.efConstruction = index_config.ef_construction

    def to_gpu(self) -> None:
        import faiss

        # clustering_index overrides the index used during clustering but leaves the quantizer on the gpu
        # https://faiss.ai/cpp_api/namespace/namespacefaiss_1_1gpu.html
        clustering_index = faiss.index_cpu_to_all_gpus(
            faiss.IndexFlat(self.module.config.embedding_dim, self.metric_type)
        )
        clustering_index.verbose = self.verbose
        index_ivf = faiss.extract_index_ivf(self.index)
        index_ivf.clustering_index = clustering_index

    def to_cpu(self) -> None:
        import faiss

        if torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu") and hasattr(faiss, "index_cpu_to_gpu"):
            self.index = faiss.index_gpu_to_cpu(self.index)

            # https://gist.github.com/mdouze/334ad6a979ac3637f6d95e9091356d3e
            # move index to cpu but leave quantizer on gpu
            index_ivf = faiss.extract_index_ivf(self.index)
            quantizer = index_ivf.quantizer
            gpu_quantizer = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, quantizer)
            index_ivf.quantizer = gpu_quantizer

    def set_verbosity(self, verbose: bool | None = None) -> None:
        import faiss

        verbose = verbose if verbose is not None else self.verbose
        index = faiss.extract_index_ivf(self.index)
        for elem in (index, index.quantizer):
            setattr(elem, "verbose", verbose)


class FaissPQIndexer(_FaissTrainIndexer):

    INDEX_FACTORY = "OPQ{num_subquantizers},PQ{num_subquantizers}x{n_bits}"

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissPQIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, module, verbose)
        self.index_config: FaissPQIndexConfig

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass


class FaissIVFPQIndexer(FaissIVFIndexer):
    INDEX_FACTORY = "OPQ{num_subquantizers},IVF{num_centroids}_HNSW32,PQ{num_subquantizers}x{n_bits}"

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissIVFPQIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        import faiss

        super().__init__(index_dir, index_config, module, verbose)
        self.index_config: FaissIVFPQIndexConfig

        index_ivf = faiss.extract_index_ivf(self.index)
        index_ivf.make_direct_map()

    def set_verbosity(self, verbose: bool | None = None) -> None:
        super().set_verbosity(verbose)
        import faiss

        verbose = verbose if verbose is not None else self.verbose
        index_ivf_pq = faiss.downcast_index(self.index.index)
        for elem in (
            index_ivf_pq.pq,
            index_ivf_pq.quantizer,
        ):
            setattr(elem, "verbose", verbose)


class FaissIndexConfig(IndexConfig):
    SUPPORTED_MODELS = {ColConfig.model_type, DprConfig.model_type}
    indexer_class: Type[Indexer] = FaissIndexer


class FaissFlatIndexConfig(FaissIndexConfig):
    indexer_class = FaissFlatIndexer


class _FaissTrainIndexConfig(FaissIndexConfig):

    indexer_class = _FaissTrainIndexer

    def __init__(self, num_train_embeddings: int | None = None) -> None:
        super().__init__()
        self.num_train_embeddings = num_train_embeddings


class FaissIVFIndexConfig(_FaissTrainIndexConfig):
    indexer_class = FaissIVFIndexer

    def __init__(
        self,
        num_train_embeddings: int | None = None,
        num_centroids: int = 262144,
        ef_construction: int = 40,
    ) -> None:
        super().__init__(num_train_embeddings)
        self.num_centroids = num_centroids
        self.ef_construction = ef_construction


class FaissPQIndexConfig(_FaissTrainIndexConfig):
    indexer_class = FaissPQIndexer

    def __init__(self, num_train_embeddings: int | None = None, num_subquantizers: int = 16, n_bits: int = 8) -> None:
        super().__init__(num_train_embeddings)
        self.num_subquantizers = num_subquantizers
        self.n_bits = n_bits


class FaissIVFPQIndexConfig(FaissIVFIndexConfig):
    indexer_class = FaissIVFPQIndexer

    def __init__(
        self,
        num_train_embeddings: int | None = None,
        num_centroids: int = 262144,
        ef_construction: int = 40,
        num_subquantizers: int = 16,
        n_bits: int = 8,
    ) -> None:
        super().__init__(num_train_embeddings, num_centroids, ef_construction)
        self.num_subquantizers = num_subquantizers
        self.n_bits = n_bits
