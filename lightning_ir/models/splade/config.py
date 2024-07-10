from typing import Literal

from ...bi_encoder.config import BiEncoderConfig


class SpladeConfig(BiEncoderConfig):
    model_type = "splade"

    def __init__(
        self,
        similarity_function: Literal["cosine", "dot"] = "dot",
        query_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "max",
        doc_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "max",
        projection: Literal["linear", "linear_no_bias", "mlm"] | None = "mlm",
        sparsification: Literal["relu", "relu_log"] | None = "relu_log",
        embedding_dim: int = 30522,
        **kwargs,
    ) -> None:
        super().__init__(
            similarity_function=similarity_function,
            query_expansion=False,
            attend_to_query_expanded_tokens=False,
            query_pooling_strategy=query_pooling_strategy,
            query_mask_scoring_tokens=None,
            doc_expansion=False,
            attend_to_doc_expanded_tokens=False,
            doc_pooling_strategy=doc_pooling_strategy,
            doc_mask_scoring_tokens=None,
            query_aggregation_function="sum",
            normalize=False,
            add_marker_tokens=False,
            embedding_dim=embedding_dim,
            projection=None if projection == "mlm" else projection,
            sparsification=sparsification,
            **kwargs,
        )
        self.projection = projection
