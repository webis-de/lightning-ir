"""
Model module for cross-encoder models.

This module defines the model class used to implement cross-encoder models.
"""

from dataclasses import dataclass
from typing import Type

import torch
from transformers import BatchEncoding

from ..base import LightningIRModel, LightningIROutput
from ..base.model import batch_encoding_wrapper
from . import CrossEncoderConfig


@dataclass
class CrossEncoderOutput(LightningIROutput):
    """Dataclass containing the output of a cross-encoder model"""

    embeddings: torch.Tensor | None = None
    """Joint query-document embeddings"""


class CrossEncoderModel(LightningIRModel):
    config_class: Type[CrossEncoderConfig] = CrossEncoderConfig
    """Configuration class for cross-encoder models."""

    def __init__(self, config: CrossEncoderConfig, *args, **kwargs):
        """A cross-encoder model that jointly encodes a query and document(s). The contextualized embeddings are
        aggragated into a single vector and fed to a linear layer which computes a final relevance score.

        :param config: Configuration for the cross-encoder model
        :type config: CrossEncoderConfig
        """
        super().__init__(config, *args, **kwargs)
        self.config: CrossEncoderConfig
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
