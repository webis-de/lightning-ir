"""
Configuration module for bi-encoder models.

This module defines the configuration class used to instantiate bi-encoder models.
"""

from typing import Any, Literal, Sequence

from ..base import LightningIRConfig


class BiEncoderConfig(LightningIRConfig):
    """Configuration class for a bi-encoder model."""

    model_type: str = "bi-encoder"
    """Model type for bi-encoder models."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalization: Literal["l2"] | None = None,
        sparsification: Literal["relu", "relu_log"] | None = None,
        add_marker_tokens: bool = False,
        **kwargs,
    ):
        """A bi-encoder model encodes queries and documents separately and computes a relevance score based on the
        similarity of the query and document embeddings. Normalization and sparsification can be applied to the
        embeddings before computing the similarity score.

        Args:
            query_length (int): Maximum query length. Defaults to 32.
            doc_length (int): Maximum document length. Defaults to 512.
            similarity_function (Literal['cosine', 'dot']): Similarity function to compute scores between query and
                document embeddings. Defaults to "dot".
            normalization (Literal['l2'] | None): Whether to normalize query and document embeddings. Defaults to None.
            sparsification (Literal['relu', 'relu_log'] | None): Whether and which sparsification function to apply.
                Defaults to None.
            add_marker_tokens (bool): Whether to prepend extra marker tokens [Q] / [D] to queries / documents.
                Defaults to False.
        """
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.similarity_function = similarity_function
        self.normalization = normalization
        self.sparsification = sparsification
        self.add_marker_tokens = add_marker_tokens
        self.embedding_dim: int | None = getattr(self, "hidden_size", None)

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from the configuration that correspond to the default config attributes for
        better readability, while always retaining the `config` attribute from the class. Serializes to a
        Python dictionary.

        Returns:
            dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        diff_dict = super().to_diff_dict()
        diff_dict.pop("embedding_dim", None)  # Exclude embedding_dim from diff_dict
        return diff_dict


class SingleVectorBiEncoderConfig(BiEncoderConfig):
    """Configuration class for a single-vector bi-encoder model."""

    model_type: str = "single-vector-bi-encoder"
    """Model type for single-vector bi-encoder models."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalization: Literal["l2"] | None = None,
        sparsification: Literal["relu", "relu_log"] | None = None,
        add_marker_tokens: bool = False,
        query_pooling_strategy: Literal["first", "mean", "max", "sum"] = "mean",
        doc_pooling_strategy: Literal["first", "mean", "max", "sum"] = "mean",
        **kwargs,
    ):
        """Configuration class for a single-vector bi-encoder model. A single-vector bi-encoder model pools the
        representations of queries and documents into a single vector before computing a similarity score.

        Args:
            query_length (int): Maximum query length. Defaults to 32.
            doc_length (int): Maximum document length. Defaults to 512.
            similarity_function (Literal['cosine', 'dot']): Similarity function to compute scores between query and
                document embeddings. Defaults to "dot".
            normalization (Literal['l2'] | None): Whether to normalize query and document embeddings. Defaults to None.
            sparsification (Literal['relu', 'relu_log'] | None): Whether and which sparsification function to apply.
                Defaults to None.
            add_marker_tokens (bool): Whether to prepend extra marker tokens [Q] / [D] to queries / documents.
                Defaults to False.
            query_pooling_strategy (Literal['first', 'mean', 'max', 'sum'] | str): How to pool the query
                token embeddings. Defaults to "mean".
            doc_pooling_strategy (Literal['first', 'mean', 'max', 'sum'] | str): How to pool document
                token embeddings. Defaults to "mean".
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            normalization=normalization,
            sparsification=sparsification,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
        self.query_pooling_strategy = query_pooling_strategy
        self.doc_pooling_strategy = doc_pooling_strategy


class MultiVectorBiEncoderConfig(BiEncoderConfig):
    """Configuration class for a multi-vector bi-encoder model."""

    model_type: str = "multi-vector-bi-encoder"
    """Model type for multi-vector bi-encoder models."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalization: Literal["l2"] | None = None,
        sparsification: None | Literal["relu", "relu_log"] = None,
        add_marker_tokens: bool = False,
        query_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        query_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "sum",
        doc_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "max",
        **kwargs,
    ):
        """A multi-vector bi-encoder model keeps the representation of all tokens in query or document and computes a
        relevance score by aggregating the similarities of query-document token pairs. Optionally, some tokens can be
        masked out during scoring.

        Args:
            query_length (int): Maximum query length. Defaults to 32.
            doc_length (int): Maximum document length. Defaults to 512.
            similarity_function (Literal['cosine', 'dot']): Similarity function to compute scores between query and
                document embeddings. Defaults to "dot".
            normalization (Literal['l2'] | None): Whether to normalize query and document embeddings. Defaults to None.
            sparsification (Literal['relu', 'relu_log'] | None): Whether and which sparsification function to apply.
                Defaults to None.
            add_marker_tokens (bool): Whether to prepend extra marker tokens [Q] / [D] to queries / documents.
                Defaults to False.
            query_mask_scoring_tokens (Sequence[str] | Literal['punctuation'] | None): Whether and which query tokens
                to ignore during scoring. Defaults to None.
            doc_mask_scoring_tokens (Sequence[str] | Literal['punctuation'] | None): Whether and which document tokens
                to ignore during scoring. Defaults to None.
            query_aggregation_function (Literal['sum', 'mean', 'max', 'harmonic_mean']): How to aggregate similarity
                scores over query tokens. Defaults to "sum".
            doc_aggregation_function (Literal['sum', 'mean', 'max', 'harmonic_mean']): How to aggregate similarity
                scores over doc tokens. Defaults to "max".
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            normalization=normalization,
            sparsification=sparsification,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
        self.query_mask_scoring_tokens = query_mask_scoring_tokens
        self.doc_mask_scoring_tokens = doc_mask_scoring_tokens
        self.query_aggregation_function = query_aggregation_function
        self.doc_aggregation_function = doc_aggregation_function
