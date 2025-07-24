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

        Args:
            query_length (int): Maximum query length. Defaults to 32.
            doc_length (int): Maximum document length. Defaults to 512.
            pooling_strategy (Literal['first', 'mean', 'max', 'sum']): Pooling strategy to aggregate the contextualized
                embeddings into a single vector for computing a relevance score. Defaults to "first".
            linear_bias (bool): Whether to use a bias in the prediction linear layer. Defaults to False.
        """
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.pooling_strategy = pooling_strategy
        self.linear_bias = linear_bias
