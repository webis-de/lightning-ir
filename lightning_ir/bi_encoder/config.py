from typing import Literal

from ..base import LightningIRConfig
from .tokenizer import BiEncoderTokenizer


class BiEncoderConfig(LightningIRConfig):
    model_type = "bi-encoder"
    tokenizer_class = BiEncoderTokenizer

    TOKENIZER_ARGS = LightningIRConfig.TOKENIZER_ARGS.union(
        {
            "query_expansion",
            "attend_to_query_expanded_tokens",
            "doc_expansion",
            "attend_to_doc_expanded_tokens",
            "add_marker_tokens",
        }
    )

    ADDED_ARGS = LightningIRConfig.ADDED_ARGS.union(
        {
            "similarity_function",
            "aggregation_function",
            "normalize",
            "embedding_dim",
            "linear",
            "linear_bias",
        }
    )

    def __init__(
        self,
        similarity_function: Literal["cosine", "l2", "dot"] = "dot",
        query_expansion: bool = False,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        attend_to_doc_expanded_tokens: bool = False,
        normalize: bool = True,
        add_marker_tokens: bool = True,
        embedding_dim: int = 128,
        linear: bool = True,
        linear_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.similarity_function = similarity_function
        self.query_expansion = query_expansion
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens
        self.normalize = normalize
        self.add_marker_tokens = add_marker_tokens
        self.embedding_dim = embedding_dim
        self.linear = linear
        self.linear_bias = linear_bias


class MultiVectorBiEncoderConfig(BiEncoderConfig):
    model_type = "multi-vector-bi-encoder"

    REMOVED_ARGS = ["pooling_strategy"]

    def __init__(
        self,
        aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "sum",
        **kwargs,
    ):
        for kwarg in self.REMOVED_ARGS:
            if kwarg in kwargs:
                kwargs.pop(kwarg)
        super().__init__(pooling_strategy=None, **kwargs)
        self.aggregation_function = aggregation_function


class SingleVectorBiEncoderConfig(BiEncoderConfig):
    model_type = "single-vector-bi-encoder"

    REMOVED_ARGS = ["aggregation_function"]

    def __init__(self, **kwargs):
        for kwarg in self.REMOVED_ARGS:
            if kwarg in kwargs:
                kwargs.pop(kwarg)
        super().__init__(aggregation_function="sum", **kwargs)
