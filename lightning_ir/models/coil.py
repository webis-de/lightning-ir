"""Configuration, model, and embedding for COIL (Contextualized Inverted List) type models. Originally proposed in
`COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List \
<https://arxiv.org/abs/2104.07186>`_."""

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from transformers import BatchEncoding

from ..bi_encoder import BiEncoderEmbedding, MultiVectorBiEncoderConfig, MultiVectorBiEncoderModel


@dataclass
class CoilEmbedding(BiEncoderEmbedding):
    """Dataclass containing embeddings and the encoding for COIL models."""

    embeddings: torch.Tensor
    """Raw embeddings of the COIL model. Should not be used directly for scoring."""
    token_embeddings: torch.Tensor | None = None
    """Token embeddings of the COIL model."""
    cls_embeddings: torch.Tensor | None = None
    """Separate [CLS] token embeddings."""


class CoilConfig(MultiVectorBiEncoderConfig):
    """Configuration class for COIL models."""

    model_type = "coil"
    """Model type for COIL models."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalize: bool = False,
        add_marker_tokens: bool = False,
        token_embedding_dim: int = 32,
        cls_embedding_dim: int = 768,
        projection: Literal["linear", "linear_no_bias"] = "linear",
        **kwargs,
    ) -> None:
        """A COIL model encodes queries and documents separately, and computes a similarity score using the maximum
        similarity ...

        Args:
            query_length (int, optional): Maximum query length in number of tokens. Defaults to 32.
            doc_length (int, optional): Maximum document length in number of tokens. Defaults to 512.
            similarity_function (Literal["cosine", "dot"]): Similarity function to compute scores between query and
                document embeddings. Defaults to "dot".
            normalize (bool): Whether to normalize query and document embeddings. Defaults to False.
            add_marker_tokens (bool): Whether to add extra marker tokens [Q] / [D] to queries / documents.
                Defaults to False.
            token_embedding_dim (int, optional): The output embedding dimension for tokens. Defaults to 32.
            cls_embedding_dim (int, optional): The output embedding dimension for the [CLS] token. Defaults to 768.
            projection (Literal["linear", "linear_no_bias"], optional): Whether and how to project the embeddings.
                Defaults to "linear".
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            normalize=normalize,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
        self.projection = projection
        self.token_embedding_dim = token_embedding_dim
        self.cls_embedding_dim = cls_embedding_dim


class CoilModel(MultiVectorBiEncoderModel):
    """Multi-vector COIL model. See :class:`.CoilConfig` for configuration options."""

    config_class = CoilConfig
    """Configuration class for COIL models."""

    def __init__(self, config: CoilConfig, *args, **kwargs) -> None:
        """Initializes a COIL model given a :class:`.CoilConfig` configuration.

        Args:
            config (CoilConfig): Configuration for the COIL model.
        """
        super().__init__(config, *args, **kwargs)
        self.config: CoilConfig
        self.token_projection = torch.nn.Linear(
            self.config.hidden_size,
            self.config.token_embedding_dim,
            bias="no_bias" not in self.config.projection,
        )
        self.cls_projection = torch.nn.Linear(
            self.config.hidden_size,
            self.config.cls_embedding_dim,
            bias="no_bias" not in self.config.projection,
        )

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> CoilEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings and scoring mask.

        Args:
            encoding (BatchEncoding): Tokenizer encodings for the text sequence.
            input_type (Literal["query", "doc"]): Type of input, either "query" or "doc".
        Returns:
            BiEncoderEmbedding: Embeddings and scoring mask.
        """
        embeddings = self._backbone_forward(**encoding).last_hidden_state

        cls_embeddings = self.cls_projection(embeddings[:, [0]])
        token_embeddings = self.token_projection(embeddings[:, 1:])

        if self.config.normalize:
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, dim=-1)
            token_embeddings = torch.nn.functional.normalize(token_embeddings, dim=-1)

        scoring_mask = self.scoring_mask(encoding, input_type)
        return CoilEmbedding(
            embeddings, scoring_mask, encoding, cls_embeddings=cls_embeddings, token_embeddings=token_embeddings
        )

    def score(
        self,
        query_embeddings: CoilEmbedding,
        doc_embeddings: CoilEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Compute relevance scores between queries and documents.

        Args:
            query_embeddings (CoilEmbedding): CLS embeddings, token embeddings, and scoring mask for the queries.
            doc_embeddings (CoilEmbedding): CLS embeddings, token embeddings, and scoring mask for the documents.
            num_docs (Sequence[int] | int | None): Specifies how many documents are passed per query. If a sequence of
                integers, `len(num_doc)` should be equal to the number of queries and `sum(num_docs)` equal to the
                number of documents, i.e., the sequence contains one value per query specifying the number of documents
                for that query. If an integer, assumes an equal number of documents per query. If None, tries to infer
                the number of documents by dividing the number of documents by the number of queries. Defaults to None.
        Returns:
            torch.Tensor: Relevance scores."""
        if query_embeddings.scoring_mask is None or doc_embeddings.scoring_mask is None:
            raise ValueError("Scoring masks expected for scoring multi-vector embeddings")
        if (
            query_embeddings.cls_embeddings is None
            or doc_embeddings.cls_embeddings is None
            or query_embeddings.token_embeddings is None
            or doc_embeddings.token_embeddings is None
        ):
            raise ValueError("COIL embeddings must contain cls_embeddings and token_embeddings")

        cls_scores = self.compute_similarity(
            BiEncoderEmbedding(query_embeddings.cls_embeddings),
            BiEncoderEmbedding(doc_embeddings.cls_embeddings),
            num_docs,
        ).view(-1)

        token_similarity = self.compute_similarity(
            BiEncoderEmbedding(query_embeddings.token_embeddings),
            BiEncoderEmbedding(doc_embeddings.token_embeddings),
            num_docs,
        )
        num_docs_t = self._parse_num_docs(
            query_embeddings.embeddings.shape[0], doc_embeddings.embeddings.shape[0], num_docs, query_embeddings.device
        )
        query = query_embeddings.encoding.input_ids.repeat_interleave(num_docs_t, 0)[:, 1:]
        docs = doc_embeddings.encoding.input_ids[:, 1:]
        mask = (query[:, :, None] == docs[:, None, :]).to(token_similarity)
        token_similarity = token_similarity * mask
        token_scores = self.aggregate_similarity(
            token_similarity, query_embeddings.scoring_mask[:, 1:], doc_embeddings.scoring_mask[:, 1:], num_docs
        )

        return cls_scores + token_scores
