"""
Model implementation for mono cross-encoder models. Originally introduced in
`Passage Re-ranking with BERT
<https://arxiv.org/abs/1901.04085>`_.
"""

from typing import Type

import torch
from transformers import BatchEncoding

from ..base.model import batch_encoding_wrapper
from ..cross_encoder.cross_encoder_config import CrossEncoderConfig
from ..cross_encoder.cross_encoder_model import CrossEncoderModel, CrossEncoderOutput


class MonoConfig(CrossEncoderConfig):
    """Configuration class for mono cross-encoder models."""

    model_type = "mono"
    """Model type for mono cross-encoder models."""

    def __init__(self, *args, **kwargs):
        """Initialize the configuration for mono cross-encoder models."""
        super().__init__(*args, **kwargs)


class MonoModel(CrossEncoderModel):
    config_class: Type[MonoConfig] = MonoConfig
    """Configuration class for mono cross-encoder models."""

    def __init__(self, config: MonoConfig, *args, **kwargs):
        """A cross-encoder model that jointly encodes a query and document(s). The contextualized embeddings are
        aggragated into a single vector and fed to a linear layer which computes a final relevance score.

        :param config: Configuration for the cross-encoder model
        :type config: CrossEncoderConfig
        """
        super().__init__(config, *args, **kwargs)
        self.linear = torch.nn.Linear(config.hidden_size, 1, bias=config.linear_bias)

    @batch_encoding_wrapper
    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        """Computes contextualized embeddings for the joint query-document input sequence and computes a relevance
        score.

        :param encoding: Tokenizer encoding for the joint query-document input sequence
        :type encoding: BatchEncoding
        :return: Output of the model
        :rtype: CrossEncoderOutput
        """
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.pooling(
            embeddings, encoding.get("attention_mask", None), pooling_strategy=self.config.pooling_strategy
        )
        scores = self.linear(embeddings).view(-1)

        return CrossEncoderOutput(scores=scores, embeddings=embeddings)
