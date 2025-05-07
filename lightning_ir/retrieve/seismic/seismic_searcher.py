from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Tuple

import torch

from lightning_ir.retrieve.base.packed_tensor import PackedTensor

try:
    _seismic_available = True
    from seismic import PySeismicIndex
except ImportError:
    _seismic_available = False
    PySeismicIndex = None

from ...bi_encoder.bi_encoder_model import BiEncoderEmbedding
from ...models import SpladeConfig
from ..base.searcher import ApproximateSearchConfig, ApproximateSearcher
from .seismic_format import SeismicFormatConverter

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderModule


class SeismicSearcher(ApproximateSearcher):
    def __init__(
        self,
        index_dir: Path | str,
        search_config: "SeismicSearchConfig",
        module: BiEncoderModule,
        use_gpu: bool = False,
    ) -> None:
        super().__init__(index_dir, search_config, module, use_gpu)
        if not _seismic_available:
            raise ImportError(
                "Please install the seismic package to use the SeismicIndexer. "
                "Instructions can be found at "
                "https://github.com/TusKANNy/seismic?tab=readme-ov-file#using-the-python-interface"
            )
        assert PySeismicIndex is not None
        self.index = PySeismicIndex.load(str(self.index_dir / ".index.seismic"))

        self.search_config: SeismicSearchConfig

    def _candidate_retrieval(self, query_embeddings: BiEncoderEmbedding) -> Tuple[PackedTensor, PackedTensor]:
        if query_embeddings.scoring_mask is None:
            embeddings = query_embeddings.embeddings[:, 0]
        else:
            embeddings = query_embeddings.embeddings[query_embeddings.scoring_mask]

        tmp_file = tempfile.NamedTemporaryFile("wb", delete_on_close=False)
        tmp_file.write((embeddings.shape[0]).to_bytes(4, byteorder="little", signed=False))
        tmp_file.write(SeismicFormatConverter.convert_to_seismic_format(embeddings))
        tmp_file.close()

        results = self.index.batch_search(
            tmp_file.name,
            k=self.search_config.k,
            query_cut=self.search_config.query_cut,
            heap_factor=self.search_config.heap_factor,
            num_threads=self.search_config.num_threads,
        )

        scores_list = []
        candidate_idcs_list = []
        num_docs = []
        for result in results:
            for score, doc_idx in result:
                scores_list.append(score)
                candidate_idcs_list.append(doc_idx)
            num_docs.append(len(result))

        scores = torch.tensor(scores_list)
        candidate_idcs = torch.tensor(candidate_idcs_list, device=query_embeddings.device)

        return PackedTensor(scores, lengths=num_docs), PackedTensor(candidate_idcs, lengths=num_docs)

    def _gather_doc_embeddings(self, idcs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Gathering doc embeddings is not supported for SeismicSearcher")


class SeismicSearchConfig(ApproximateSearchConfig):

    search_class = SeismicSearcher
    SUPPORTED_MODELS = {SpladeConfig.model_type}

    def __init__(
        self,
        k: int = 10,
        candidate_k: int = 100,
        imputation_strategy: Literal["min", "gather", "zero"] = "min",
        query_cut: int = 10,
        heap_factor: float = 0.7,
        num_threads: int = 1,
    ) -> None:
        if imputation_strategy == "gather":
            raise ValueError("Imputation strategy 'gather' is not supported for SeismicSearcher")
        super().__init__(k=k, candidate_k=candidate_k, imputation_strategy=imputation_strategy)
        self.query_cut = query_cut
        self.heap_factor = heap_factor
        self.num_threads = num_threads
