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

    ALLOW_BATCHING = True

    def __init__(self, config: CrossEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: CrossEncoderConfig
        self.linear = torch.nn.Linear(config.hidden_size, 1, bias=config.linear_bias)

    def batched_backbone_forward(self, encoding: BatchEncoding) -> torch.Tensor:
        if not self.ALLOW_BATCHING:
            return self.backbone_forward(encoding)
        batch_size = encoding["input_ids"].shape[0]
        outputs = []
        sub_encoding = encoding
        while True:
            try:
                outputs.append(self.backbone_forward(sub_encoding).last_hidden_state)
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    batch_size = batch_size // 2
                    if batch_size == 0:
                        raise e
                    sub_encoding = BatchEncoding(**{key: value[:batch_size] for key, value in sub_encoding.items()})
                else:
                    raise e
        if batch_size == encoding["input_ids"].shape[0]:
            return outputs[0]
        num_batches = encoding["input_ids"].shape[0] // batch_size - 1
        for i in range(1, num_batches):
            sub_encoding = BatchEncoding(
                **{key: value[batch_size * i : batch_size * (i + 1)] for key, value in encoding.items()}
            )
            outputs.append(self.backbone_forward(sub_encoding).last_hidden_state)
        return torch.cat(outputs)

    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        embeddings = self.batched_backbone_forward(**encoding)
        embeddings = self._pooling(
            embeddings, encoding.get("attention_mask", None), pooling_strategy=self.config.pooling_strategy
        )
        scores = self.linear(embeddings).view(-1)
        return CrossEncoderOutput(scores=scores, embeddings=embeddings)
