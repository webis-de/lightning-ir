"""
Configuration module for cross-encoder models.

This module defines the configuration class used to instantiate cross-encoder models.
"""

from typing import Literal

from ..base import LightningIRConfig


class CrossEncoderConfig(LightningIRConfig):
    model_type: str = "cross-encoder"
    """Model type for cross-encoder models."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        pooling_strategy: Literal["first", "mean", "max", "sum"] = "first",
        linear_bias: bool = False,
        **kwargs
    ):
        """Configuration class for a cross-encoder model

        :param query_length: Maximum query length, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length, defaults to 512
        :type doc_length: int, optional
        :param pooling_strategy: Pooling strategy to aggregate the contextualized embeddings into a single vector for
            computing a relevance score, defaults to "first"
        :type pooling_strategy: Literal['first', 'mean', 'max', 'sum'], optional
        :param linear_bias: Whether to use a bias in the prediction linear layer, defaults to False
        :type linear_bias: bool, optional
        """
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.pooling_strategy = pooling_strategy
        self.linear_bias = linear_bias
