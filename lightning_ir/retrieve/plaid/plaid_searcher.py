"""Plaid Searcher using fast-plaid library for Lightning IR Framework"""

from pathlib import Path
from typing import List, Tuple

import torch
from fast_plaid import search

from ...bi_encoder import BiEncoderModule, BiEncoderOutput
from ...models import ColConfig
from ..base.packed_tensor import PackedTensor
from ..base.searcher import SearchConfig, Searcher


class PlaidSearcher(Searcher):
    """Searcher for Plaid using fast-plaid library."""

    def __init__(
        self,
        index_dir: Path,
        search_config: "PlaidSearchConfig",
        module: BiEncoderModule,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the PlaidSearcher.

        Args:
            index_dir (Path | str): Directory where the Plaid index is stored.
            search_config (PlaidSearchConfig): Configuration for the Plaid searcher.
            module (BiEncoderModule): The BiEncoder module used for searching.
            use_gpu (bool): Whether to use GPU for searching. Defaults to False.
        """
        super().__init__(index_dir, search_config, module, use_gpu)
        self.search_config: PlaidSearchConfig

        self.load()

        # with open(self.index_dir / "doclens.0.json", "r") as doc_lens_f:
        #     self.doc_lengths = json.load(doc_lens_f)

    def load(self) -> None:
        """Load the Plaid index from the specified directory."""

        self.index = search.FastPlaid(
            index=str(self.index_dir), device=self.device.type, preload_index=self.search_config.preload_index
        )

    def search(self, output: BiEncoderOutput) -> Tuple[PackedTensor, List[List[str]]]:
        """Search for relevant documents using the Plaid index.

        Args:
            output (BiEncoderOutput): The output from the BiEncoder module containing query embeddings.
        Returns:
            Tuple[PackedTensor, List[List[str]]]: A tuple containing the scores and the corresponding document IDs.
        Raises:
            ValueError: If the output does not contain query embeddings.
            ValueError: If the index is not loaded. Call load() before searching.
        """
        if output.query_embeddings is None:
            raise ValueError("Expected query_embeddings in BiEncoderOutput")

        if not self.index:
            raise ValueError("Index not loaded. Call load() before searching.")

        scores = self.index.search(
            queries_embeddings=output.query_embeddings.embeddings.detach(),
            top_k=self.search_config.k,
        )
        all_doc_ids = []
        all_scores = []
        for result in scores:
            doc_indices, doc_scores = zip(*result)
            doc_ids = [self.doc_ids[idx] for idx in doc_indices]
            all_doc_ids.append(doc_ids)
            all_scores.append(list(doc_scores))

        lengths = [len(doc_scores) for doc_scores in all_scores]
        flat_scores = [score for sublist in all_scores for score in sublist]
        return PackedTensor(torch.tensor(flat_scores), lengths=lengths), all_doc_ids


class PlaidSearchConfig(SearchConfig):
    """Configuration class for Plaid searchers in the Lightning IR framework."""

    search_class = PlaidSearcher
    SUPPORTED_MODELS = {ColConfig.model_type}

    def __init__(
        self,
        k: int,
        candidate_k: int = 256,
        n_cells: int = 1,
        centroid_score_threshold: float = 0.5,
        preload_index: bool = True,
    ) -> None:
        """Initialize the PlaidSearchConfig.

        Args:
            k (int): Number of top documents to retrieve.
            candidate_k (int): Number of candidate documents to consider for scoring. Defaults to 256.
            n_cells (int): Number of cells to use for centroid retrieval. Defaults to 1.
            centroid_score_threshold (float): Threshold for filtering candidates based on centroid scores.
                Defaults to 0.5.
        """
        super().__init__(k)
        self.candidate_k = candidate_k
        self.n_cells = n_cells
        self.centroid_score_threshold = centroid_score_threshold
        self.preload_index = preload_index
