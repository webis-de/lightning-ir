"""SeismicIndexer class for indexing documents using the Seismic library."""

import os
from pathlib import Path

import numpy as np
import torch

try:
    _seismic_available = True
    import seismic
    from seismic import SeismicDataset, SeismicIndex

    STRING_TYPE = seismic.get_seismic_string()
except ImportError:
    STRING_TYPE = None
    _seismic_available = False
    SeismicIndex = SeismicDataset = None


from ...bi_encoder import BiEncoderModule, BiEncoderOutput
from ...data import IndexBatch
from ...models import SpladeConfig
from ..base import IndexConfig, Indexer


class SeismicIndexer(Indexer):
    """Indexer for Seismic, a residual-based indexing method for efficient retrieval."""

    def __init__(
        self,
        index_dir: Path,
        index_config: "SeismicIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the SeismicIndexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (SeismicIndexConfig): Configuration for the Seismic indexer.
            module (BiEncoderModule): The BiEncoder module used for indexing.
            verbose (bool): Whether to print verbose output during indexing. Defaults to False.
        Raises:
            ImportError: If the seismic package is not available.
        """
        super().__init__(index_dir, index_config, module, verbose)
        if _seismic_available is False:
            raise ImportError(
                "Please install the seismic package to use the SeismicIndexer. "
                "Instructions can be found at "
                "https://github.com/TusKANNy/seismic?tab=readme-ov-file#using-the-python-interface"
            )
        self.index_config: SeismicIndexConfig
        assert SeismicDataset is not None
        self.seismic_dataset = SeismicDataset()

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        """Add embeddings from the index batch to the Seismic index.

        Args:
            index_batch (IndexBatch): Batch of data containing embeddings to be indexed.
            output (BiEncoderOutput): Output from the BiEncoder module containing embeddings.
        Raises:
            ValueError: If the output does not contain document embeddings.
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

        num_docs = len(index_batch.doc_ids)
        self.doc_ids.extend(index_batch.doc_ids)
        self.doc_lengths.extend(doc_lengths.int().cpu().tolist())
        self.num_embeddings += embeddings.shape[0]
        self.num_docs += num_docs

        for idx, doc_id in enumerate(index_batch.doc_ids):
            non_zero = embeddings[idx].nonzero().view(-1)
            values = embeddings[idx][non_zero].float().numpy(force=True)
            tokens = np.array(self.module.tokenizer.convert_ids_to_tokens(non_zero), dtype="U30")
            self.seismic_dataset.add_document(doc_id, tokens, values)

    def save(self) -> None:
        """Save the Seismic index to disk."""
        super().save()

        assert SeismicIndex is not None
        index = SeismicIndex.build_from_dataset(
            self.seismic_dataset,
            n_postings=self.index_config.num_postings,
            centroid_fraction=self.index_config.centroid_fraction,
            min_cluster_size=self.index_config.min_cluster_size,
            summary_energy=self.index_config.summary_energy,
            nknn=self.index_config.num_k_nearest_neighbors,
            batched_indexing=self.index_config.batch_size,
            num_threads=self.index_config.num_threads,
        )
        index.save(str(self.index_dir) + os.path.sep)


class SeismicIndexConfig(IndexConfig):
    """Configuration for the Seismic indexer."""

    indexer_class = SeismicIndexer
    SUPPORTED_MODELS = {SpladeConfig.model_type}

    def __init__(
        self,
        num_postings: int = 3_500,
        centroid_fraction: float = 0.1,
        min_cluster_size: int = 2,
        summary_energy: float = 0.4,
        num_k_nearest_neighbors: int = 0,
        batch_size: int | None = None,
        num_threads: int = 0,
    ) -> None:
        """Initialize the SeismicIndexConfig.

        Args:
            num_postings (int): Number of postings to keep in the index. Defaults to 3500.
            centroid_fraction (float): Fraction of centroids to keep. Defaults to 0.1.
            min_cluster_size (int): Minimum size of clusters. Defaults to 2.
            summary_energy (float): Energy threshold for summaries. Defaults to 0.4.
            num_k_nearest_neighbors (int): Number of nearest neighbors to consider. Defaults to 0.
            batch_size (int | None): Batch size for indexing. Defaults to None.
            num_threads (int): Number of threads to use for indexing. Defaults to 0.
        """
        super().__init__()
        self.num_postings = num_postings
        self.centroid_fraction = centroid_fraction
        self.summary_energy = summary_energy
        self.min_cluster_size = min_cluster_size
        self.num_k_nearest_neighbors = num_k_nearest_neighbors
        self.batch_size = batch_size
        self.num_threads = num_threads
