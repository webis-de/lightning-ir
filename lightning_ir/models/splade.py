from typing import Literal

from ..bi_encoder import BiEncoderConfig, BiEncoderModel


class SpladeConfig(BiEncoderConfig):
    model_type = "splade"

    def __init__(
        self,
        query_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "max",
        doc_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "max",
        projection: Literal["linear", "linear_no_bias", "mlm"] | None = "mlm",
        sparsification: Literal["relu", "relu_log"] | None = "relu_log",
        embedding_dim: int = 30522,
        **kwargs,
    ) -> None:
        kwargs["query_expansion"] = False
        kwargs["attend_to_query_expanded_tokens"] = False
        kwargs["query_mask_scoring_tokens"] = None
        kwargs["doc_expansion"] = False
        kwargs["attend_to_doc_expanded_tokens"] = False
        kwargs["doc_mask_scoring_tokens"] = None
        kwargs["query_aggregation_function"] = "sum"
        kwargs["normalize"] = False
        kwargs["add_marker_tokens"] = False
        super().__init__(
            query_pooling_strategy=query_pooling_strategy,
            doc_pooling_strategy=doc_pooling_strategy,
            embedding_dim=embedding_dim,
            projection=projection,
            sparsification=sparsification,
            **kwargs,
        )


class SpladeModel(BiEncoderModel):
    config_class = SpladeConfig
