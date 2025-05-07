"""Configuration, model, and tokenizer for Col (Contextualized Late Interaction) type models. Originally proposed in
`ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT \
<https://dl.acm.org/doi/abs/10.1145/3397271.3401075>`_ as the ColBERT model. This implementation generalizes the model
to work with any transformer backbone model.
"""

from typing import Literal, Sequence

import torch
from transformers import BatchEncoding

from ..bi_encoder import BiEncoderEmbedding, BiEncoderTokenizer, MultiVectorBiEncoderConfig, MultiVectorBiEncoderModel


class ColConfig(MultiVectorBiEncoderConfig):
    """Configuration class for a Col model."""

    model_type = "col"
    """Model type for a Col model."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalize: bool = False,
        add_marker_tokens: bool = False,
        query_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        query_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "sum",
        doc_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "max",
        embedding_dim: int = 128,
        projection: Literal["linear", "linear_no_bias"] = "linear",
        query_expansion: bool = False,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        attend_to_doc_expanded_tokens: bool = False,
        **kwargs,
    ):
        """A Col model encodes queries and documents separately and computes a late interaction score between the query
        and document embeddings. The aggregation behavior of the late-interaction function can be parameterized with
        the `aggregation_function` arguments. The dimensionality of the token embeddings is down-projected using a
        linear layer. Queries and documents can optionally be expanded with mask tokens. Optionally, a set of tokens can
        be ignored during scoring.

        :param query_length: Maximum query length, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length, defaults to 512
        :type doc_length: int, optional
        :param similarity_function: Similarity function to compute scores between query and document embeddings,
            defaults to "dot"
        :type similarity_function: Literal['cosine', 'dot'], optional
        :param normalize: Whether to normalize query and document embeddings, defaults to False
        :type normalize: bool, optional
        :param add_marker_tokens: Whether to add extra marker tokens [Q] / [D] to queries / documents, defaults to False
        :type add_marker_tokens: bool, optional
        :param query_mask_scoring_tokens: Whether and which query tokens to ignore during scoring, defaults to None
        :type query_mask_scoring_tokens: Sequence[str] | Literal['punctuation'] | None, optional
        :param doc_mask_scoring_tokens: Whether and which document tokens to ignore during scoring, defaults to None
        :type doc_mask_scoring_tokens: Sequence[str] | Literal['punctuation'] | None, optional
        :param query_aggregation_function: How to aggregate similarity scores over query tokens, defaults to "sum"
        :type query_aggregation_function: Literal[ 'sum', 'mean', 'max', 'harmonic_mean' ], optional
        :param doc_aggregation_function: How to aggregate similarity scores over document tokens, defaults to "max"
        :type doc_aggregation_function: Literal[ 'sum', 'mean', 'max', 'harmonic_mean' ], optional
        :param embedding_dim: The output embedding dimension, defaults to 768
        :type embedding_dim: int, optional
        :param projection: Whether and how to project the output emeddings, defaults to "linear"
        :type projection: Literal['linear', 'linear_no_bias', 'mlm'] | None, optional
        :param query_expansion: Whether to expand queries with mask tokens, defaults to False
        :type query_expansion: bool, optional
        :param attend_to_query_expanded_tokens: Whether to allow query tokens to attend to mask tokens,
            defaults to False
        :type attend_to_query_expanded_tokens: bool, optional
        :param doc_expansion: Whether to expand documents with mask tokens, defaults to False
        :type doc_expansion: bool, optional
        :param attend_to_doc_expanded_tokens: Whether to allow document tokens to attend to mask tokens,
            defaults to False
        :type attend_to_doc_expanded_tokens: bool, optional
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            add_marker_tokens=add_marker_tokens,
            query_mask_scoring_tokens=query_mask_scoring_tokens,
            doc_mask_scoring_tokens=doc_mask_scoring_tokens,
            query_aggregation_function=query_aggregation_function,
            doc_aggregation_function=doc_aggregation_function,
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.projection = projection
        self.query_expansion = query_expansion
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens
        self.normalize = normalize
        self.add_marker_tokens = add_marker_tokens


class ColModel(MultiVectorBiEncoderModel):
    """Multi-vector late-interaction Col model. See :class:`.ColConfig` for configuration options."""

    config_class = ColConfig
    """Configuration class for the Col model."""

    def __init__(self, config: ColConfig, *args, **kwargs) -> None:
        """Initializes a Col model given a :class:`.ColConfig`.

        :param config: Configuration for the Col model
        :type config: ColConfig
        """
        super().__init__(config, *args, **kwargs)
        if config.embedding_dim is None:
            raise ValueError("Embedding dimension must be specified in the configuration.")
        self.projection = torch.nn.Linear(
            config.hidden_size, config.embedding_dim, bias="no_bias" not in config.projection
        )

    def scoring_mask(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> torch.Tensor:
        """Computes a scoring mask for batched tokenized text sequences which is used in the scoring function to mask
        out vectors during scoring.

        :param encoding: Tokenizer encodings for the text sequence
        :type encoding: BatchEncoding
        :param input_type: Type of input, either "query" or "doc"
        :type input_type: Literal["query", "doc"]
        :return: Scoring mask
        :rtype: torch.Tensor
        """
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        scoring_mask = attention_mask
        expansion = getattr(self.config, f"{input_type}_expansion")
        if expansion or scoring_mask is None:
            scoring_mask = torch.ones_like(input_ids, dtype=torch.bool)
        scoring_mask = scoring_mask.bool()
        mask_scoring_input_ids = getattr(self, f"{input_type}_mask_scoring_input_ids")
        if mask_scoring_input_ids is not None:
            ignore_mask = input_ids[..., None].eq(mask_scoring_input_ids.to(input_ids.device)).any(-1)
            scoring_mask = scoring_mask & ~ignore_mask
        return scoring_mask

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings and scoring mask.

        :param encoding: Tokenizer encodings for the text sequence
        :type encoding: BatchEncoding
        :param input_type: Type of input, either "query" or "doc"
        :type input_type: Literal["query", "doc"]
        :return: Embeddings and scoring mask
        :rtype: BiEncoderEmbedding
        """
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.projection(embeddings)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self.scoring_mask(encoding, input_type)
        return BiEncoderEmbedding(embeddings, scoring_mask, encoding)


class ColTokenizer(BiEncoderTokenizer):
    """:class:`.LightningIRTokenizer` for Col models."""

    config_class = ColConfig
    """Configuration class for the tokenizer."""

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        add_marker_tokens: bool = False,
        query_expansion: bool = False,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        attend_to_doc_expanded_tokens: bool = False,
        **kwargs,
    ):
        """Initializes a Col model's tokenizer. Encodes queries and documents separately. Optionally adds marker tokens
        to encoded input sequences and expands queries and documents with mask tokens.

        :param query_length: Maximum query length in number of tokens, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length in number of tokens, defaults to 512
        :type doc_length: int, optional
        :param add_marker_tokens: Whether to add marker tokens to the query and document input sequences,
            defaults to False
        :type add_marker_tokens: bool, optional
        :param query_expansion: Whether to expand queries with mask tokens, defaults to False
        :type query_expansion: bool, optional
        :param attend_to_query_expanded_tokens: Whether to let non-expanded query tokens be able to attend to mask
            expanded query tokens, defaults to False
        :type attend_to_query_expanded_tokens: bool, optional
        :param doc_expansion: Whether to expand documents with mask tokens, defaults to False
        :type doc_expansion: bool, optional
        :param attend_to_doc_expanded_tokens: Whether to let non-expanded document tokens be able to attend to
            mask expanded document tokens, defaults to False
        :type attend_to_doc_expanded_tokens: bool, optional
        :raises ValueError: If add_marker_tokens is True and a non-supported tokenizer is used
        """
        super().__init__(
            *args,
            query_length=query_length,
            doc_length=doc_length,
            query_expansion=query_expansion,
            attend_to_query_expanded_tokens=attend_to_query_expanded_tokens,
            doc_expansion=doc_expansion,
            attend_to_doc_expanded_tokens=attend_to_doc_expanded_tokens,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
        self.query_expansion = query_expansion
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens

    def _expand(self, encoding: BatchEncoding, attend_to_expanded_tokens: bool) -> BatchEncoding:
        """Applies mask expansion to the input encoding."""
        input_ids = encoding["input_ids"]
        input_ids[input_ids == self.pad_token_id] = self.mask_token_id
        encoding["input_ids"] = input_ids
        if attend_to_expanded_tokens:
            encoding["attention_mask"].fill_(1)
        return encoding

    def tokenize_input_sequence(
        self, text: Sequence[str] | str, input_type: Literal["query", "doc"], *args, **kwargs
    ) -> BatchEncoding:
        """Tokenizes an input sequence. This method is used to tokenize both queries and documents.

        :param queries: Single string or multiple strings to tokenize
        :type queries: Sequence[str] | str
        :return: Tokenized input sequences
        :rtype: BatchEncoding
        """
        expansion = getattr(self, f"{input_type}_expansion")
        if expansion:
            kwargs["padding"] = "max_length"
        encoding = super().tokenize_input_sequence(text, input_type, *args, **kwargs)
        if self.query_expansion:
            encoding = self._expand(encoding, self.attend_to_query_expanded_tokens)
        return encoding
