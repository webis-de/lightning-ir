import os
import tempfile
from pathlib import Path

import torch

try:
    _seismic_available = True
    from seismic import PySeismicIndex
except ImportError:
    _seismic_available = False
    PySeismicIndex = None


from ...bi_encoder import BiEncoderConfig, BiEncoderOutput
from ...data import IndexBatch
from ...models import SpladeConfig
from ..base import IndexConfig, Indexer
from .seismic_format import SeismicFormatConverter


class SeismicIndexer(Indexer):
    def __init__(
        self,
        index_dir: Path,
        index_config: "SeismicIndexConfig",
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, bi_encoder_config, verbose)
        if _seismic_available is False:
            raise ImportError(
                "Please install the seismic package to use the SeismicIndexer. "
                "Instructions can be found at "
                "https://github.com/TusKANNy/seismic?tab=readme-ov-file#using-the-python-interface"
            )
        self.index_config: SeismicIndexConfig
        self.tmp_file = tempfile.NamedTemporaryFile("wb", delete_on_close=False)
        # NOTE overwrite the number of documents when saving
        self.tmp_file.write((0).to_bytes(4, byteorder="little", signed=False))

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

        num_docs = len(index_batch.doc_ids)
        self.doc_ids.extend(index_batch.doc_ids)
        self.doc_lengths.extend(doc_lengths.int().cpu().tolist())
        self.num_embeddings += embeddings.shape[0]
        self.num_docs += num_docs

        self.tmp_file.write(SeismicFormatConverter.convert_to_seismic_format(embeddings))

    def save(self) -> None:
        super().save()

        self.tmp_file.seek(0)
        self.tmp_file.write((self.num_docs).to_bytes(4, byteorder="little", signed=False))

        self.tmp_file.close()
        assert PySeismicIndex is not None
        index = PySeismicIndex.build(
            self.tmp_file.name,
            n_postings=self.index_config.num_postings,
            centroid_fraction=self.index_config.centroid_fraction,
            truncated_kmeans_training=self.index_config.truncated_kmeans_training,
            truncation_size=self.index_config.truncation_size,
            min_cluster_size=self.index_config.min_cluster_size,
            summary_energy=self.index_config.summary_energy,
        )
        index.save(str(self.index_dir) + os.path.sep)


class SeismicIndexConfig(IndexConfig):
    indexer_class = SeismicIndexer
    SUPPORTED_MODELS = {SpladeConfig.model_type}

    def __init__(
        self,
        num_postings: int = 4_000,
        centroid_fraction: float = 0.1,
        summary_energy: float = 0.4,
        truncated_kmeans_training: bool = False,
        truncation_size: int = 16,
        min_cluster_size: int = 2,
    ) -> None:
        super().__init__()
        self.num_postings = num_postings
        self.centroid_fraction = centroid_fraction
        self.summary_energy = summary_energy
        self.truncated_kmeans_training = truncated_kmeans_training
        self.truncation_size = truncation_size
        self.min_cluster_size = min_cluster_size
