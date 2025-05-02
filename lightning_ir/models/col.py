from typing import Literal, Sequence

import torch
from transformers import BatchEncoding

from ..bi_encoder import BiEncoderConfig, BiEncoderModel, BiEncoderEmbedding


class ColConfig(BiEncoderConfig):
    model_type = "col"

    def __init__(
        self,
        query_expansion: bool = True,
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = "punctuation",
        embedding_dim: int = 128,
        projection: Literal["linear", "linear_no_bias"] | None = "linear_no_bias",
        **kwargs,
    ) -> None:
        kwargs["query_pooling_strategy"] = None
        kwargs["doc_expansion"] = False
        kwargs["attend_to_doc_expanded_tokens"] = False
        kwargs["doc_pooling_strategy"] = None
        super().__init__(
            query_expansion=query_expansion,
            doc_mask_scoring_tokens=doc_mask_scoring_tokens,
            embedding_dim=embedding_dim,
            projection=projection,
            **kwargs,
        )


class ColModel(BiEncoderModel):
    config_class = ColConfig

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        expansion = True if input_type == "query" else False  # getattr(self.config, f"{input_type}_expansion")
        pooling_strategy = None
        projection = self.projection if self.config.tie_projection else getattr(self, f"{input_type}_projection")
        mask_scoring_input_ids = getattr(self, f"{input_type}_mask_scoring_input_ids")

        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = projection(embeddings)
        embeddings = self._sparsification(embeddings, self.config.sparsification)
        embeddings = self._pooling(embeddings, encoding["attention_mask"], pooling_strategy)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self.scoring_mask(encoding, expansion, pooling_strategy, mask_scoring_input_ids)
        return BiEncoderEmbedding(embeddings, scoring_mask, encoding)
