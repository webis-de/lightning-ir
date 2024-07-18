from typing import Literal, Sequence

from ...bi_encoder.config import BiEncoderConfig


class ColConfig(BiEncoderConfig):
    model_type = "col"

    def __init__(
        self,
        similarity_function: Literal["cosine", "dot"] = "dot",
        query_expansion: bool = True,
        attend_to_query_expanded_tokens: bool = False,
        query_mask_scoring_tokens: Sequence[str] | None = None,
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = "punctuation",
        query_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "sum",
        normalize: bool = False,
        add_marker_tokens: bool = False,
        embedding_dim: int = 128,
        projection: Literal["linear", "linear_no_bias"] | None = "linear_no_bias",
        **kwargs,
    ) -> None:
        kwargs["query_pooling_strategy"] = None
        kwargs["doc_expansion"] = False
        kwargs["attend_to_doc_expanded_tokens"] = False
        kwargs["doc_pooling_strategy"] = None
        super().__init__(
            similarity_function=similarity_function,
            query_expansion=query_expansion,
            attend_to_query_expanded_tokens=attend_to_query_expanded_tokens,
            query_mask_scoring_tokens=query_mask_scoring_tokens,
            doc_mask_scoring_tokens=doc_mask_scoring_tokens,
            query_aggregation_function=query_aggregation_function,
            normalize=normalize,
            add_marker_tokens=add_marker_tokens,
            embedding_dim=embedding_dim,
            projection=projection,
            **kwargs,
        )
