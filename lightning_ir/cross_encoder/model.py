from dataclasses import dataclass

import torch
from transformers import BatchEncoding

from ..base import LightningIRModel, LightningIROutput
from . import CrossEncoderConfig


@dataclass
class CrossEncoderOutput(LightningIROutput):
    embeddings: torch.Tensor | None = None


class CrossEncoderModel(LightningIRModel):
    config_class = CrossEncoderConfig

    def __init__(self, config: CrossEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: CrossEncoderConfig
        self.linear = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        embeddings = self.backbone_forward(**encoding).last_hidden_state
        embeddings = self._pooling(
            embeddings, encoding.get("attention_mask", None), pooling_strategy=self.config.pooling_strategy
        )
        scores = self.linear(embeddings).squeeze(-1)
        return CrossEncoderOutput(scores=scores, embeddings=embeddings)
