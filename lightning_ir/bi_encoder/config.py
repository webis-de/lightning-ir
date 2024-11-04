"""
Configuration module for bi-encoder models.

This module defines the configuration class used to instantiate bi-encoder models.
"""

import json
import os
from os import PathLike
from typing import Any, Dict, Literal, Sequence, Tuple

from ..base import LightningIRConfig


class BiEncoderConfig(LightningIRConfig):
    model_type: str = "bi-encoder"
    """Model type for bi-encoder models."""

    TOKENIZER_ARGS = LightningIRConfig.TOKENIZER_ARGS.union(
        {
            "query_expansion",
            "attend_to_query_expanded_tokens",
            "doc_expansion",
            "attend_to_doc_expanded_tokens",
            "add_marker_tokens",
        }
    )
    """Arguments for the tokenizer."""

    ADDED_ARGS = LightningIRConfig.ADDED_ARGS.union(
        {
            "similarity_function",
            "query_pooling_strategy",
            "query_mask_scoring_tokens",
            "query_aggregation_function",
            "doc_pooling_strategy",
            "doc_mask_scoring_tokens",
            "normalize",
            "sparsification",
            "embedding_dim",
            "projection",
        }
    ).union(TOKENIZER_ARGS)
    """Arguments added to the configuration."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        query_expansion: bool = False,
        attend_to_query_expanded_tokens: bool = False,
        query_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "mean",
        query_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        query_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "sum",
        doc_expansion: bool = False,
        attend_to_doc_expanded_tokens: bool = False,
        doc_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "mean",
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        normalize: bool = False,
        sparsification: Literal["relu", "relu_log"] | None = None,
        add_marker_tokens: bool = False,
        embedding_dim: int = 768,
        projection: Literal["linear", "linear_no_bias", "mlm"] | None = "linear",
        **kwargs,
    ):
        """Configuration class for a bi-encoder model.

        :param query_length: Maximum query length, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length, defaults to 512
        :type doc_length: int, optional
        :param similarity_function: Similarity function to compute scores between query and document embeddings,
            defaults to "dot"
        :type similarity_function: Literal['cosine', 'dot'], optional
        :param query_expansion: Whether to expand queries with mask tokens, defaults to False
        :type query_expansion: bool, optional
        :param attend_to_query_expanded_tokens: Whether to allow query tokens to attend to mask tokens,
            defaults to False
        :type attend_to_query_expanded_tokens: bool, optional
        :param query_pooling_strategy: Whether and how to pool the query token embeddings, defaults to "mean"
        :type query_pooling_strategy: Literal['first', 'mean', 'max', 'sum'] | None, optional
        :param query_mask_scoring_tokens: Whether and which query tokens to ignore during scoring, defaults to None
        :type query_mask_scoring_tokens: Sequence[str] | Literal['punctuation'] | None, optional
        :param query_aggregation_function: How to aggregate similarity scores over query tokens, defaults to "sum"
        :type query_aggregation_function: Literal[ 'sum', 'mean', 'max', 'harmonic_mean' ], optional
        :param doc_expansion: Whether to expand documents with mask tokens, defaults to False
        :type doc_expansion: bool, optional
        :param attend_to_doc_expanded_tokens: Whether to allow document tokens to attend to mask tokens,
            defaults to False
        :type attend_to_doc_expanded_tokens: bool, optional
        :param doc_pooling_strategy: Whether andhow to pool document token embeddings, defaults to "mean"
        :type doc_pooling_strategy: Literal['first', 'mean', 'max', 'sum'] | None, optional
        :param doc_mask_scoring_tokens: Whether and which document tokens to ignore during scoring, defaults to None
        :type doc_mask_scoring_tokens: Sequence[str] | Literal['punctuation'] | None, optional
        :param normalize: Whether to normalize query and document embeddings, defaults to False
        :type normalize: bool, optional
        :param sparsification: Whether and which sparsification function to apply, defaults to None
        :type sparsification: Literal['relu', 'relu_log'] | None, optional
        :param add_marker_tokens: Whether to add extra marker tokens [Q] / [D] to queries / documents, defaults to False
        :type add_marker_tokens: bool, optional
        :param embedding_dim: The output embedding dimension, defaults to 768
        :type embedding_dim: int, optional
        :param projection: Whether and how to project the output emeddings, defaults to "linear"
        :type projection: Literal['linear', 'linear_no_bias', 'mlm'] | None, optional
        """
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.similarity_function = similarity_function
        self.query_expansion = query_expansion
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.query_pooling_strategy = query_pooling_strategy
        self.query_mask_scoring_tokens = query_mask_scoring_tokens
        self.query_aggregation_function = query_aggregation_function
        self.doc_expansion = doc_expansion
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens
        self.doc_pooling_strategy = doc_pooling_strategy
        self.doc_mask_scoring_tokens = doc_mask_scoring_tokens
        self.normalize = normalize
        self.sparsification = sparsification
        self.add_marker_tokens = add_marker_tokens
        self.embedding_dim = embedding_dim
        self.projection = projection

    def to_dict(self) -> Dict[str, Any]:
        """Overrides the transformers.PretrainedConfig.to_dict_ method to include the added arguments, the backbone
        model type, and remove the mask scoring tokens.

        .. _transformers.PretrainedConfig.to_dict: \
https://huggingface.co/docs/transformers/en/main_classes/configuration#transformers.PretrainedConfig.to_dict

        :return: Configuration dictionary
        :rtype: Dict[str, Any]
        """

        output = super().to_dict()
        if "query_mask_scoring_tokens" in output:
            output.pop("query_mask_scoring_tokens")
        if "doc_mask_scoring_tokens" in output:
            output.pop("doc_mask_scoring_tokens")
        return output

    def save_pretrained(self, save_directory: str | PathLike, **kwargs) -> None:
        """Overrides the transformers.PretrainedConfig.save_pretrained_ method to addtionally save the tokens which
        should be maksed during scoring.

        .. _transformers.PretrainedConfig.save_pretrained: \
https://huggingface.co/docs/transformers/en/main_classes/configuration#transformers.PretrainedConfig.save_pretrained

        :param save_directory: Directory to save the configuration
        :type save_directory: str | PathLike
        """
        with open(os.path.join(save_directory, "mask_scoring_tokens.json"), "w") as f:
            json.dump({"query": self.query_mask_scoring_tokens, "doc": self.doc_mask_scoring_tokens}, f)
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | PathLike, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Overrides the transformers.PretrainedConfig.get_config_dict_ method to load the tokens that should be masked
        during scoring.

        .. _transformers.PretrainedConfig.get_config_dict: \
https://huggingface.co/docs/transformers/en/main_classes/configuration#transformers.PretrainedConfig.get_config_dict

        :param pretrained_model_name_or_path: Name or path of the pretrained model
        :type pretrained_model_name_or_path: str | PathLike
        :return: Configuration dictionary and additional keyword arguments
        :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
        """
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        mask_scoring_tokens = None
        mask_scoring_tokens_path = os.path.join(pretrained_model_name_or_path, "mask_scoring_tokens.json")
        if os.path.exists(mask_scoring_tokens_path):
            with open(mask_scoring_tokens_path) as f:
                mask_scoring_tokens = json.load(f)
            config_dict["query_mask_scoring_tokens"] = mask_scoring_tokens["query"]
            config_dict["doc_mask_scoring_tokens"] = mask_scoring_tokens["doc"]
        return config_dict, kwargs
