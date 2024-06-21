from typing import Literal, Sequence

from ...bi_encoder.config import BiEncoderConfig


class ColConfig(BiEncoderConfig):
    model_type = "col"

    def __init__(
        self,
        similarity_function: Literal["cosine", "l2", "dot"] = "dot",
        query_expansion: bool = True,
        attend_to_query_expanded_tokens: bool = False,
        query_mask_scoring_tokens: Sequence[str] | None = None,
        doc_mask_scoring_tokens: (
            Sequence[str] | Literal["punctuation"] | None
        ) = "punctuation",
        doc_aggregation_function: Literal[
            "sum", "mean", "max", "harmonic_mean"
        ] = "sum",
        normalize: bool = True,
        add_marker_tokens: bool = True,
        embedding_dim: int = 128,
        linear: bool = True,
        linear_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            similarity_function=similarity_function,
            query_expansion=query_expansion,
            attend_to_query_expanded_tokens=attend_to_query_expanded_tokens,
            query_pooling_strategy=None,
            query_mask_scoring_tokens=query_mask_scoring_tokens,
            doc_expansion=False,
            attend_to_doc_expanded_tokens=False,
            doc_pooling_strategy=None,
            doc_mask_scoring_tokens=doc_mask_scoring_tokens,
            doc_aggregation_function=doc_aggregation_function,
            normalize=normalize,
            add_marker_tokens=add_marker_tokens,
            embedding_dim=embedding_dim,
            linear=linear,
            linear_bias=linear_bias,
            **kwargs,
        )
