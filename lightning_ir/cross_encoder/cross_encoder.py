from typing import Any, Dict

from transformers import PretrainedConfig


class CrossEncoderConfig(PretrainedConfig):
    model_type = "cross-encoder"

    ADDED_ARGS = []

    TOKENIZER_ARGS = []

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def to_added_args_dict(self) -> Dict[str, Any]:
        return {
            arg: getattr(self, arg) for arg in self.ADDED_ARGS if hasattr(self, arg)
        }

    def to_tokenizer_dict(self) -> Dict[str, Any]:
        return {arg: getattr(self, arg) for arg in self.TOKENIZER_ARGS}

    @classmethod
    def from_other(
        cls,
        config: PretrainedConfig,
        **kwargs,
    ) -> "CrossEncoderConfig":
        return cls.from_dict({**config.to_dict(), **kwargs})
