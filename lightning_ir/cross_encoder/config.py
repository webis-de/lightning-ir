from typing import Literal, Type

from ..base import LightningIRConfig
from .tokenizer import CrossEncoderTokenizer


class CrossEncoderConfig(LightningIRConfig):
    model_type = "cross-encoder"
    tokenizer_class: Type[CrossEncoderTokenizer] = CrossEncoderTokenizer

    ADDED_ARGS = LightningIRConfig.ADDED_ARGS.union({"pooling_strategy"})

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "first",
        **kwargs
    ):
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.pooling_strategy = pooling_strategy
