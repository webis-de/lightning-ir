""""""

from typing import Literal, Sequence

import torch
from transformers import BatchEncoding

from ...bi_encoder import BiEncoderEmbedding, BiEncoderTokenizer, MultiVectorBiEncoderConfig, MultiVectorBiEncoderModel


class XTRConfig(MultiVectorBiEncoderConfig):
    """Configuration class for XTR model."""

    model_type = "xtr"

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalization: Literal["l2"] | None = None,
        sparsification: None | Literal["relu", "relu_log", "relu_2xlog"] = None,
        add_marker_tokens: bool = False,
        query_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        query_aggregation_function: Literal["sum", "mean", "max"] = "sum",
        doc_aggregation_function: Literal["sum", "mean", "max"] = "max",
        **kwargs,
    ) -> None:
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            normalization=normalization,
            sparsification=sparsification,
            add_marker_tokens=add_marker_tokens,
            query_mask_scoring_tokens=query_mask_scoring_tokens,
            doc_mask_scoring_tokens=doc_mask_scoring_tokens,
            query_aggregation_function=query_aggregation_function,
            doc_aggregation_function=doc_aggregation_function,
            **kwargs,
        )


class XTRModel(MultiVectorBiEncoderModel):
    """XTR bi-encoder model."""

    config_class = XTRConfig

    def __init__(self, config: XTRConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)


class XTRTokenizer(BiEncoderTokenizer):
    """Tokenizer class for XTR model."""

    config_class = XTRConfig

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        add_marker_tokens: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            query_length=query_length,
            doc_length=doc_length,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
