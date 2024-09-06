from dataclasses import dataclass
from typing import Type

import torch
from transformers import BatchEncoding

from ..base import LightningIRModel, LightningIROutput
from . import CrossEncoderConfig


@dataclass
class CrossEncoderOutput(LightningIROutput):
    embeddings: torch.Tensor | None = None


class CrossEncoderModel(LightningIRModel):
    config_class: Type[CrossEncoderConfig] = CrossEncoderConfig

    ALLOW_SUB_BATCHING = True
    """Flag to allow mini batches of documents for a single query. Set to false for listwise models to  ensure 
    correctness."""

    def __init__(self, config: CrossEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: CrossEncoderConfig
        self.linear = torch.nn.Linear(config.hidden_size, 1, bias=config.linear_bias)

    def _batched_backbone_forward(self, encoding: BatchEncoding) -> torch.Tensor:
        if not self.ALLOW_SUB_BATCHING:
            return self._backbone_forward(**encoding)
        return super()._batched_backbone_forward(encoding)

    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        embeddings = self._batched_backbone_forward(encoding)
        embeddings = self._pooling(
            embeddings, encoding.get("attention_mask", None), pooling_strategy=self.config.pooling_strategy
        )
        scores = self.linear(embeddings).view(-1)
        return CrossEncoderOutput(scores=scores, embeddings=embeddings)
