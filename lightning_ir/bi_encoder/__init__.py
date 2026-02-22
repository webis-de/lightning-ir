"""Base module for bi-encoder models.

This module provides the main classes and functions for bi-encoder models, including configurations, models,
modules, and tokenizers.

Bi-encoders are neural models that encode queries and documents separately into dense vector representations,
then compute relevance scores based on the similarity between these embeddings."""

from .bi_encoder_config import (
    BiEncoderConfig,
    MultiVectorBiEncoderConfig,
    SingleVectorBiEncoderConfig,
)
from .bi_encoder_model import (
    BiEncoderEmbedding,
    BiEncoderModel,
    BiEncoderOutput,
    MultiVectorBiEncoderModel,
    SingleVectorBiEncoderModel,
)
from .bi_encoder_module import BiEncoderModule
from .bi_encoder_tokenizer import BiEncoderTokenizer

__all__ = [
    "BiEncoderConfig",
    "BiEncoderEmbedding",
    "BiEncoderModel",
    "BiEncoderModule",
    "BiEncoderOutput",
    "BiEncoderTokenizer",
    "MultiVectorBiEncoderConfig",
    "MultiVectorBiEncoderModel",
    "SingleVectorBiEncoderConfig",
    "SingleVectorBiEncoderModel",
]
