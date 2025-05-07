from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Tuple

import torch

from ...bi_encoder.bi_encoder_model import BiEncoderEmbedding
from ...models import ColConfig, DprConfig
from ..base.packed_tensor import PackedTensor
from ..base.searcher import ApproximateSearchConfig, ApproximateSearcher

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderModule


class FaissSearcher(ApproximateSearcher):
    def __init__(
        self,
        index_dir: Path | str,
        search_config: FaissSearchConfig,
        module: BiEncoderModule,
        use_gpu: bool = False,
    ) -> None:
        import faiss

        self.search_config: FaissSearchConfig
        self.index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
        if use_gpu and hasattr(faiss, "index_cpu_to_all_gpus"):
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        ivf_index = None
        try:
            ivf_index = faiss.extract_index_ivf(self.index)
        except RuntimeError:
            pass
        if ivf_index is not None:
            ivf_index.nprobe = search_config.n_probe
            quantizer = getattr(ivf_index, "quantizer", None)
            if quantizer is not None:
                downcasted_quantizer = faiss.downcast_index(quantizer)
                hnsw = getattr(downcasted_quantizer, "hnsw", None)
                if hnsw is not None:
                    hnsw.efSearch = search_config.ef_search
        super().__init__(index_dir, search_config, module, use_gpu)

    def _candidate_retrieval(self, query_embeddings: BiEncoderEmbedding) -> Tuple[PackedTensor, PackedTensor]:
        if query_embeddings.scoring_mask is None:
            embeddings = query_embeddings.embeddings[:, 0]
        else:
            embeddings = query_embeddings.embeddings[query_embeddings.scoring_mask]
        candidate_scores, candidate_idcs = self.index.search(embeddings.float().cpu(), self.search_config.candidate_k)
        candidate_scores = torch.from_numpy(candidate_scores).view(-1)
        candidate_idcs = torch.from_numpy(candidate_idcs).view(-1)
        num_candidates_per_query_vector = [self.search_config.candidate_k] * embeddings.shape[0]
        packed_candidate_scores = PackedTensor(candidate_scores, lengths=num_candidates_per_query_vector)
        packed_candidate_idcs = PackedTensor(candidate_idcs, lengths=num_candidates_per_query_vector)
        return packed_candidate_scores, packed_candidate_idcs

    def _gather_doc_embeddings(self, idcs: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.index.reconstruct_batch(idcs))


class FaissSearchConfig(ApproximateSearchConfig):
    search_class = FaissSearcher
    SUPPORTED_MODELS = {ColConfig.model_type, DprConfig.model_type}

    def __init__(
        self,
        k: int = 10,
        candidate_k: int = 100,
        imputation_strategy: Literal["min", "gather", "zero"] = "gather",
        n_probe: int = 1,
        ef_search: int = 16,
    ) -> None:
        super().__init__(k, candidate_k, imputation_strategy)
        self.n_probe = n_probe
        self.ef_search = ef_search
