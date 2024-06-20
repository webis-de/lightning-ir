from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Type, Literal

from transformers import PretrainedConfig

if TYPE_CHECKING:
    from . import LightningIRTokenizer


class LightningIRConfig(PretrainedConfig):
    model_type = "lightning-ir"
    backbone_model_type: str | None = None
    tokenizer_class: Type[LightningIRTokenizer] | None = None

    TOKENIZER_ARGS = {"query_length", "doc_length"}
    ADDED_ARGS = TOKENIZER_ARGS.union({"pooling_strategy"})

    def __init__(
        self,
        pooling_strategy: Literal["cls", "mean", "max", "sum"] | None = "mean",
        query_length: int = 32,
        doc_length: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling_strategy = pooling_strategy
        self.query_length = query_length
        self.doc_length = doc_length

    def to_added_args_dict(self) -> Dict[str, Any]:
        return {
            arg: getattr(self, arg) for arg in self.ADDED_ARGS if hasattr(self, arg)
        }

    def to_tokenizer_dict(self) -> Dict[str, Any]:
        return {arg: getattr(self, arg) for arg in self.TOKENIZER_ARGS}

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(super(), "to_dict"):
            output = getattr(super(), "to_dict")()
        else:
            output = self.to_added_args_dict()
        if self.__class__.model_type is not None:
            output["backbone_model_type"] = self.__class__.backbone_model_type
        return output
