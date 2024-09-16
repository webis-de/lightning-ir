from dataclasses import dataclass
from typing import Type

import torch
from transformers import BatchEncoding

from ..base import LightningIRModel, LightningIROutput
from ..base.model import _batch_encoding
from . import CrossEncoderConfig


@dataclass
class CrossEncoderOutput(LightningIROutput):
    embeddings: torch.Tensor | None = None


class CrossEncoderModel(LightningIRModel):
    config_class: Type[CrossEncoderConfig] = CrossEncoderConfig

    def __init__(self, config: CrossEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: CrossEncoderConfig
        self.linear = torch.nn.Linear(config.hidden_size, 1, bias=config.linear_bias)

    @_batch_encoding
    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self._pooling(
            embeddings, encoding.get("attention_mask", None), pooling_strategy=self.config.pooling_strategy
        )
        scores = self.linear(embeddings).view(-1)
        return CrossEncoderOutput(scores=scores, embeddings=embeddings)
