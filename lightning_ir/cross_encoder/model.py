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
        self._sub_batch_size: int | None = None

    def batched_backbone_forward(self, encoding: BatchEncoding) -> torch.Tensor:
        if not self.ALLOW_BATCHING:
            return self.backbone_forward(**encoding)
        if self._sub_batch_size is None:
            self._sub_batch_size = encoding.input_ids.shape[0]
        outputs = []
        sub_encoding = encoding
        remaining_encoding = encoding
        while True:
            try:
                num_batches = -(remaining_encoding.input_ids.shape[0] // -self._sub_batch_size)
                for _ in range(num_batches):
                    sub_encoding = BatchEncoding(
                        {key: value[: self._sub_batch_size] for key, value in remaining_encoding.items()}
                    )
                    outputs.append(self.backbone_forward(**sub_encoding).last_hidden_state)
                    remaining_encoding = BatchEncoding(
                        {key: value[self._sub_batch_size :] for key, value in remaining_encoding.items()}
                    )
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "CUDACachingAllocator.cpp" in str(e):
                    self._sub_batch_size = self._sub_batch_size // 2
                    if self._sub_batch_size == 0:
                        raise e
                else:
                    raise e
        output = torch.cat(outputs)
        assert output.shape[0] == encoding.input_ids.shape[0]
        return output

    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        embeddings = self.batched_backbone_forward(encoding)
        embeddings = self._pooling(
            embeddings, encoding.get("attention_mask", None), pooling_strategy=self.config.pooling_strategy
        )
        scores = self.linear(embeddings).view(-1)
        return CrossEncoderOutput(scores=scores, embeddings=embeddings)
