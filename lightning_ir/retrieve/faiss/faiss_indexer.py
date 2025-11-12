"""FAISS Indexer for Lightning IR Framework"""

import warnings
from pathlib import Path

import torch

from ...bi_encoder import BiEncoderModule, BiEncoderOutput
from ...data import IndexBatch
from ...models import ColConfig, DprConfig
from ..base import IndexConfig, Indexer


class FaissIndexer(Indexer):
    """Base class for FAISS indexers in the Lightning IR framework."""

    INDEX_FACTORY: str

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the FaissIndexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (FaissIndexConfig): Configuration for the FAISS index.
            module (BiEncoderModule): The BiEncoderModule to use for indexing.
            verbose (bool): Whether to enable verbose output. Defaults to False.
        Raises:
            ValueError: If the similarity function is not supported.
        """
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
        """Move the FAISS index to GPU."""
        pass

    def to_cpu(self) -> None:
        """Move the FAISS index to CPU."""
        pass

    def set_verbosity(self, verbose: bool | None = None) -> None:
        """set the verbosity of the FAISS index.

        Args:
            verbose (bool | None): Whether to enable verbose output. If None, uses the index's current verbosity
                setting. Defaults to None.
        """
        self.index.verbose = self.verbose if verbose is None else verbose

    def process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Process embeddings before adding them to the FAISS index.

        Args:
            embeddings (torch.Tensor): The embeddings to process.
        Returns:
            torch.Tensor: The processed embeddings.
        """
        return embeddings

    def save(self) -> None:
        """Save the FAISS index to disk.

        Raises:
            ValueError: If the number of embeddings does not match the index's total number of entries.
        """
        super().save()
        import faiss

        if self.num_embeddings != self.index.ntotal:
            raise ValueError("number of embeddings does not match index.ntotal")
        if torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu"):
            self.index = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        """Add embeddings to the FAISS index.

        Args:
            index_batch (IndexBatch): The batch containing document indices and embeddings.
            output (BiEncoderOutput): The output from the bi-encoder module containing document embeddings.
        Raises:
            ValueError: If the document embeddings are not present in the output.
        """
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
    """FAISS Flat Indexer for exact nearest neighbor search using FAISS."""

    INDEX_FACTORY = "Flat"

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissFlatIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the FaissFlatIndexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (FaissFlatIndexConfig): Configuration for the FAISS flat index.
            module (BiEncoderModule): The BiEncoderModule to use for indexing.
            verbose (bool): Whether to enable verbose output. Defaults to False.
        """
        super().__init__(index_dir, index_config, module, verbose)
        self.index_config: FaissFlatIndexConfig

    def to_gpu(self) -> None:
        """Move the FAISS flat index to GPU."""
        pass

    def to_cpu(self) -> None:
        """Move the FAISS flat index to CPU."""
        pass


class _FaissTrainIndexer(FaissIndexer):
    """Base class for FAISS indexers that require training on embeddings before indexing."""

    INDEX_FACTORY = ""  # class only acts as mixin

    def __init__(
        self,
        index_dir: Path,
        index_config: "_FaissTrainIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the _FaissTrainIndexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (_FaissTrainIndexConfig): Configuration for the FAISS index that requires training.
            module (BiEncoderModule): The BiEncoderModule to use for indexing.
            verbose (bool): Whether to enable verbose output. Defaults to False.
        Raises:
            ValueError: If num_train_embeddings is not set in the index configuration.
        """
        super().__init__(index_dir, index_config, module, verbose)
        if index_config.num_train_embeddings is None:
            raise ValueError("num_train_embeddings must be set")
        self.num_train_embeddings = index_config.num_train_embeddings

        self._train_embeddings: torch.Tensor | None = torch.full(
            (self.num_train_embeddings, self.module.config.embedding_dim), torch.nan, dtype=torch.float32
        )

    def process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Process embeddings before adding them to the FAISS index.

        Args:
            embeddings (torch.Tensor): The embeddings to process.
        Returns:
            torch.Tensor: The processed embeddings.
        """
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
            warnings.warn(
                "Corpus contains less tokens/documents than num_train_embeddings. Removing NaN embeddings.",
                stacklevel=2,
            )
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
    """FAISS IVF Indexer for approximate nearest neighbor search using FAISS with Inverted File System (IVF)."""

    INDEX_FACTORY = "IVF{num_centroids},Flat"

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissIVFIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the FaissIVFIndexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (FaissIVFIndexConfig): Configuration for the FAISS IVF index.
            module (BiEncoderModule): The BiEncoderModule to use for indexing.
            verbose (bool): Whether to enable verbose output. Defaults to False.
        """
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
        """Move the FAISS IVF index to GPU."""
        import faiss

        # clustering_index overrides the index used during clustering but leaves the quantizer on the gpu
        # https://faiss.ai/cpp_api/namespace/namespacefaiss_1_1gpu.html
        if faiss.get_num_gpus() == 0:
            return
        clustering_index = faiss.index_cpu_to_all_gpus(
            faiss.IndexFlat(self.module.config.embedding_dim, self.metric_type)
        )
        clustering_index.verbose = self.verbose
        index_ivf = faiss.extract_index_ivf(self.index)
        index_ivf.clustering_index = clustering_index

    def to_cpu(self) -> None:
        """Move the FAISS IVF index to CPU."""
        import faiss

        if faiss.get_num_gpus() == 0:
            return
        self.index = faiss.index_gpu_to_cpu(self.index)

        # https://gist.github.com/mdouze/334ad6a979ac3637f6d95e9091356d3e
        # move index to cpu but leave quantizer on gpu
        index_ivf = faiss.extract_index_ivf(self.index)
        quantizer = index_ivf.quantizer
        gpu_quantizer = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, quantizer)
        index_ivf.quantizer = gpu_quantizer

    def set_verbosity(self, verbose: bool | None = None) -> None:
        """set the verbosity of the FAISS IVF index.

        Args:
            verbose (bool | None): Whether to enable verbose output. Defaults to None.
        """
        import faiss

        verbose = verbose if verbose is not None else self.verbose
        index = faiss.extract_index_ivf(self.index)
        for elem in (index, index.quantizer):
            elem.verbose = verbose


class FaissPQIndexer(_FaissTrainIndexer):
    """FAISS PQ Indexer for approximate nearest neighbor search using Product Quantization (PQ)."""

    INDEX_FACTORY = "OPQ{num_subquantizers},PQ{num_subquantizers}x{n_bits}"

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissPQIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the FaissPQIndexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (FaissPQIndexConfig): Configuration for the FAISS PQ index.
            module (BiEncoderModule): The BiEncoderModule to use for indexing.
            verbose (bool): Whether to enable verbose output. Defaults to False.
        """
        super().__init__(index_dir, index_config, module, verbose)
        self.index_config: FaissPQIndexConfig

    def to_gpu(self) -> None:
        """Move the FAISS PQ index to GPU."""
        pass

    def to_cpu(self) -> None:
        """Move the FAISS PQ index to CPU."""
        pass


class FaissIVFPQIndexer(FaissIVFIndexer):
    """FAISS IVFPQ Indexer for approximate nearest neighbor search using Inverted File System (IVF) with Product
    Quantization (PQ)."""

    INDEX_FACTORY = "OPQ{num_subquantizers},IVF{num_centroids}_HNSW32,PQ{num_subquantizers}x{n_bits}"

    def __init__(
        self,
        index_dir: Path,
        index_config: "FaissIVFPQIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the FaissIVFPQIndexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (FaissIVFPQIndexConfig): Configuration for the FAISS IVFPQ index.
            module (BiEncoderModule): The BiEncoderModule to use for indexing.
            verbose (bool): Whether to enable verbose output. Defaults to False.
        """
        import faiss

        super().__init__(index_dir, index_config, module, verbose)
        self.index_config: FaissIVFPQIndexConfig

        index_ivf = faiss.extract_index_ivf(self.index)
        index_ivf.make_direct_map()

    def set_verbosity(self, verbose: bool | None = None) -> None:
        """set the verbosity of the FAISS IVFPQ index.

        Args:
            verbose (bool | None): Whether to enable verbose output. Defaults to None.
        """
        super().set_verbosity(verbose)
        import faiss

        verbose = verbose if verbose is not None else self.verbose
        index_ivf_pq = faiss.downcast_index(self.index.index)
        for elem in (
            index_ivf_pq.pq,
            index_ivf_pq.quantizer,
        ):
            elem.verbose = verbose


class FaissIndexConfig(IndexConfig):
    """Configuration class for FAISS indexers in the Lightning IR framework."""

    SUPPORTED_MODELS = {ColConfig.model_type, DprConfig.model_type}
    indexer_class: type[Indexer] = FaissIndexer


class FaissFlatIndexConfig(FaissIndexConfig):
    """Configuration class for FAISS flat indexers in the Lightning IR framework."""

    indexer_class = FaissFlatIndexer


class _FaissTrainIndexConfig(FaissIndexConfig):
    """Base configuration class for FAISS indexers that require training on embeddings before indexing."""

    indexer_class = _FaissTrainIndexer

    def __init__(self, num_train_embeddings: int | None = None) -> None:
        """Initialize the _FaissTrainIndexConfig.

        Args:
            num_train_embeddings (int | None): Number of embeddings to use for training the index. If None, it will
                be set later. Defaults to None.
        """
        super().__init__()
        self.num_train_embeddings = num_train_embeddings


class FaissIVFIndexConfig(_FaissTrainIndexConfig):
    """Configuration class for FAISS IVF indexers in the Lightning IR framework."""

    indexer_class = FaissIVFIndexer

    def __init__(
        self,
        num_train_embeddings: int | None = None,
        num_centroids: int = 262144,
        ef_construction: int = 40,
    ) -> None:
        """Initialize the FaissIVFIndexConfig.

        Args:
            num_train_embeddings (int | None): Number of embeddings to use for training the index. If None, it will be
                set later. Defaults to None.
            num_centroids (int): Number of centroids for the IVF index. Defaults to 262144.
            ef_construction (int): The size of the dynamic list used during construction. Defaults to 40.
        """
        super().__init__(num_train_embeddings)
        self.num_centroids = num_centroids
        self.ef_construction = ef_construction


class FaissPQIndexConfig(_FaissTrainIndexConfig):
    """Configuration class for FAISS PQ indexers in the Lightning IR framework."""

    indexer_class = FaissPQIndexer

    def __init__(self, num_train_embeddings: int | None = None, num_subquantizers: int = 16, n_bits: int = 8) -> None:
        """Initialize the FaissPQIndexConfig.

        Args:
            num_train_embeddings (int | None): Number of embeddings to use for training the index. If None, it will
                be set later. Defaults to None.
            num_subquantizers (int): Number of subquantizers for the PQ index. Defaults to 16.
            n_bits (int): Number of bits for the PQ index. Defaults to 8.
        """
        super().__init__(num_train_embeddings)
        self.num_subquantizers = num_subquantizers
        self.n_bits = n_bits


class FaissIVFPQIndexConfig(FaissIVFIndexConfig):
    """Configuration class for FAISS IVFPQ indexers in the Lightning IR framework."""

    indexer_class = FaissIVFPQIndexer

    def __init__(
        self,
        num_train_embeddings: int | None = None,
        num_centroids: int = 262144,
        ef_construction: int = 40,
        num_subquantizers: int = 16,
        n_bits: int = 8,
    ) -> None:
        """Initialize the FaissIVFPQIndexConfig.

        Args:
            num_train_embeddings (int | None): Number of embeddings to use for training the index. If None, it will
                be set later. Defaults to None.
            num_centroids (int): Number of centroids for the IVF index. Defaults to 262144.
            ef_construction (int): The size of the dynamic list used during construction. Defaults to 40.
            num_subquantizers (int): Number of subquantizers for the PQ index. Defaults to 16.
            n_bits (int): Number of bits for the PQ index. Defaults to 8.
        """
        super().__init__(num_train_embeddings, num_centroids, ef_construction)
        self.num_subquantizers = num_subquantizers
        self.n_bits = n_bits
