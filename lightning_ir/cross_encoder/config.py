from typing import Literal

from ..base import LightningIRConfig
from .tokenizer import CrossEncoderTokenizer


class CrossEncoderConfig(LightningIRConfig):
    model_type = "cross-encoder"
    tokenizer_class = CrossEncoderTokenizer

    def __init__(
        self,
        pooling_strategy: Literal["cls", "mean", "max", "sum"] | None = "cls",
        **kwargs
    ):
        super().__init__(pooling_strategy=pooling_strategy, **kwargs)
