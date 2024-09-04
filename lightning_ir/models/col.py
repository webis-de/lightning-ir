from typing import Literal, Sequence

from ..bi_encoder import BiEncoderConfig, BiEncoderModel


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
