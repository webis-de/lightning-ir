from typing import Literal

from ..base import LightningIRConfig


class CrossEncoderConfig(LightningIRConfig):
    model_type = "cross-encoder"

    ADDED_ARGS = LightningIRConfig.ADDED_ARGS.union({"pooling_strategy", "linear_bias"})

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        pooling_strategy: Literal["first", "mean", "max", "sum"] = "first",
        linear_bias: bool = False,
        **kwargs
    ):
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.pooling_strategy = pooling_strategy
        self.linear_bias = linear_bias
