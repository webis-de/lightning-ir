"""
Model module for cross-encoder models.

This module defines the model class used to implement cross-encoder models.
"""

from abc import ABC, abstractmethod
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


class CrossEncoderModel(LightningIRModel, ABC):
    config_class: Type[CrossEncoderConfig] = CrossEncoderConfig
    """Configuration class for cross-encoder models."""

    def __init__(self, config: CrossEncoderConfig, *args, **kwargs):
        """A cross-encoder model that jointly encodes a query and document(s). The contextualized embeddings are
        aggragated into a single vector and fed to a linear layer which computes a final relevance score.

        Args:
            config (CrossEncoderConfig): Configuration for the cross-encoder model.
        """
        super().__init__(config, *args, **kwargs)
        self.config: CrossEncoderConfig

    @batch_encoding_wrapper
    @abstractmethod
    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        """Computes contextualized embeddings for the joint query-document input sequence and computes a relevance
        score.

        Args:
            encoding (BatchEncoding): Tokenizer encoding for the joint query-document input sequence.
        Returns:
            CrossEncoderOutput: Output of the model.
        """
        pass
