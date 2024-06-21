import json
import os
from os import PathLike
from typing import Any, Dict, Literal, Sequence, Tuple, Type

from ..base import LightningIRConfig
from .tokenizer import BiEncoderTokenizer


class BiEncoderConfig(LightningIRConfig):
    model_type = "bi-encoder"
    tokenizer_class: Type[BiEncoderTokenizer] = BiEncoderTokenizer

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
            "query_pooling_strategy",
            "doc_pooling_strategy",
            "doc_aggregation_function",
            "normalize",
            "embedding_dim",
            "linear",
            "linear_bias",
            "mask_scoring_tokens",
        }
    )

    def __init__(
        self,
        similarity_function: Literal["cosine", "l2", "dot"] = "dot",
        query_expansion: bool = False,
        attend_to_query_expanded_tokens: bool = False,
        query_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "mean",
        query_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        doc_expansion: bool = False,
        attend_to_doc_expanded_tokens: bool = False,
        doc_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "mean",
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        doc_aggregation_function: Literal[
            "sum", "mean", "max", "harmonic_mean"
        ] = "sum",
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
        self.query_pooling_strategy = query_pooling_strategy
        self.query_mask_scoring_tokens = query_mask_scoring_tokens
        self.doc_expansion = doc_expansion
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens
        self.doc_pooling_strategy = doc_pooling_strategy
        self.doc_mask_scoring_tokens = doc_mask_scoring_tokens
        self.doc_aggregation_function = doc_aggregation_function
        self.normalize = normalize
        self.add_marker_tokens = add_marker_tokens
        self.embedding_dim = embedding_dim
        self.linear = linear
        self.linear_bias = linear_bias

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        if "query_mask_scoring_tokens" in output:
            output.pop("query_mask_scoring_tokens")
        if "doc_mask_scoring_tokens" in output:
            output.pop("doc_mask_scoring_tokens")
        return output

    def save_pretrained(
        self, save_directory: str | PathLike, push_to_hub: bool = False, **kwargs
    ):
        with open(os.path.join(save_directory, "mask_scoring_tokens.json"), "w") as f:
            json.dump(
                {
                    "query": self.query_mask_scoring_tokens,
                    "doc": self.doc_mask_scoring_tokens,
                },
                f,
            )
        return super().save_pretrained(save_directory, push_to_hub, **kwargs)

    @classmethod
    def get_config_dict(
        first, pretrained_model_name_or_path: str | PathLike, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        config_dict, kwargs = super().get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        mask_scoring_tokens = None
        mask_scoring_tokens_path = os.path.join(
            pretrained_model_name_or_path, "mask_scoring_tokens.json"
        )
        if os.path.exists(mask_scoring_tokens_path):
            with open(mask_scoring_tokens_path) as f:
                mask_scoring_tokens = json.load(f)
            config_dict["query_mask_scoring_tokens"] = mask_scoring_tokens["query"]
            config_dict["doc_mask_scoring_tokens"] = mask_scoring_tokens["doc"]
        return config_dict, kwargs
