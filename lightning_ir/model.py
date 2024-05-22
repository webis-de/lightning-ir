from typing import Any, Dict

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


class LightningIRConfig(PretrainedConfig):
    model_type = "bi-encoder"

    ADDED_ARGS = []

    TOKENIZER_ARGS = []

    def __init__(self, **kwargs):
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
    ) -> "LightningIRConfig":
        return cls.from_dict({**config.to_dict(), **kwargs})


@dataclass
class LightningIROutput(ModelOutput):
    scores: torch.Tensor | None = None


class LightningIRModel(PreTrainedModel):
    def __init__(self, config: LightningIRConfig):
        super().__init__(config)

    def forward(self, *args, **kwargs) -> LightningIROutput:
        raise NotImplementedError
