"""Base module for cross-encoder models.

This module provides the main classes and functions for cross-encoder models, including configurations, models,
modules, and tokenizers.

Cross-encoders are neural models that jointly encode queries and documents into a single representation, then compute
relevance scores based on this combined representation.
"""

from .cross_encoder_config import CrossEncoderConfig
from .cross_encoder_model import CrossEncoderModel, CrossEncoderOutput
from .cross_encoder_module import CrossEncoderModule
from .cross_encoder_tokenizer import CrossEncoderTokenizer

__all__ = [
    "CrossEncoderConfig",
    "CrossEncoderModel",
    "CrossEncoderModule",
    "CrossEncoderOutput",
    "CrossEncoderTokenizer",
]
