from typing import Any, Dict

import torch
from transformers import PretrainedConfig

from ..model import LightningIRConfig, LightningIRModel


class CrossEncoderConfig(LightningIRConfig):
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


class CrossEncoderModel(LightningIRModel):
    def __init__(self, config: CrossEncoderConfig, encoder_module_name: str):
        super().__init__(config)
        self.config: CrossEncoderConfig
        self.encoder_module_name = encoder_module_name
        self.linear = torch.nn.Linear(self.config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state[:, 0]
        output = self.linear(output)
        return output.squeeze(-1)

    @property
    def encoder(self):
        return getattr(self, self.encoder_module_name)
