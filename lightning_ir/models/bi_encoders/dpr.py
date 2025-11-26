"""Configuration and model for DPR (Dense Passage Retriever) type models. Originally proposed in \
`Dense Passage Retrieval for Open-Domain Question Answering \
<https://arxiv.org/abs/2004.04906>`_. This model type is also known as a SentenceTransformer model:
`Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks \
<https://arxiv.org/abs/1908.10084>`_.
"""

from typing import Literal

import torch
from transformers import BatchEncoding

from ...bi_encoder import BiEncoderEmbedding, SingleVectorBiEncoderConfig, SingleVectorBiEncoderModel
from ...modeling_utils.embedding_post_processing import Pooler, Sparsifier


class DprConfig(SingleVectorBiEncoderConfig):
    """Configuration class for a DPR model."""

    model_type = "lir-dpr"
    """Model type for a DPR model."""

    def __init__(
        self,
        query_length: int | None = 32,
        doc_length: int | None = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalization_strategy: Literal["l2"] | None = None,
        sparsification_strategy: Literal["relu", "relu_log", "relu_2xlog"] | None = None,
        add_marker_tokens: bool = False,
        pooling_strategy: Literal["first", "mean", "max", "sum"] = "first",
        embedding_dim: int | None = None,
        projection: Literal["linear", "linear_no_bias"] | None = "linear",
        **kwargs,
    ) -> None:
        """A DPR model encodes queries and documents separately. Before computing the similarity score, the
        contextualized token embeddings are aggregated to obtain a single embedding using a pooling strategy.
        Optionally, the pooled embeddings can be projected using a linear layer.

        Args:
            query_length (int | None): Maximum number of tokens per query. If None does not truncate. Defaults to 32.
            doc_length (int | None): Maximum number of tokens per document. If None does not truncate. Defaults to 512.
            similarity_function (Literal["cosine", "dot"]): Similarity function to compute scores between query and
                document embeddings. Defaults to "dot".
            normalization_strategy (Literal['l2'] | None): Whether to normalization_strategy query and document
                embeddings.
                Defaults to None.
            sparsification_strategy (Literal['relu', 'relu_log', 'relu_2xlog'] | None): Whether and which
                sparsification_strategy function to apply. Defaults to None.
            add_marker_tokens (bool): Whether to add marker tokens to the input sequences. Defaults to False.
            pooling_strategy (Literal["first", "mean", "max", "sum"]): Pooling strategy for query and document
                embeddings. Defaults to "first".
            embedding_dim (int | None): Dimension of the final embeddings. If None, it will be set to the hidden size
                of the backbone model. Defaults to None.
            projection (Literal["linear", "linear_no_bias"] | None): type of projection layer to apply on the pooled
                embeddings. If None, no projection is applied. Defaults to "linear".
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            normalization_strategy=normalization_strategy,
            sparsification_strategy=sparsification_strategy,
            add_marker_tokens=add_marker_tokens,
            pooling_strategy=pooling_strategy,
            **kwargs,
        )
        hidden_size = getattr(self, "hidden_size", None)
        if projection is None:
            self.embedding_dim = hidden_size
        else:
            self.embedding_dim = embedding_dim or hidden_size
        self.projection = projection


class DprModel(SingleVectorBiEncoderModel):
    """A single-vector DPR model. See :class:`DprConfig` for configuration options."""

    config_class = DprConfig
    """Configuration class for a DPR model."""

    def __init__(self, config: SingleVectorBiEncoderConfig, *args, **kwargs) -> None:
        """Initializes a DPR model given a :class:`DprConfig`.

        Args:
            config (SingleVectorBiEncoderConfig): Configuration for the DPR model.
        Raises:
            ValueError: If the embedding dimension is not specified in the configuration.
        """
        super().__init__(config, *args, **kwargs)
        if self.config.projection is None:
            self.projection: torch.nn.Module = torch.nn.Identity()
        else:
            if self.config.embedding_dim is None:
                raise ValueError("Unable to determine embedding dimension.")
            self.projection = torch.nn.Linear(
                self.config.hidden_size,
                self.config.embedding_dim,
                bias="no_bias" not in self.config.projection,
            )
        self.pooler = Pooler(config)
        self.sparsifier = Sparsifier(config)

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings and scoring mask.

        Args:
            encoding (BatchEncoding): Tokenizer encodings for the text sequence.
            input_type (Literal["query", "doc"]): type of input, either "query" or "doc".
        Returns:
            BiEncoderEmbedding: Embeddings and scoring mask.
        """
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.pooler(embeddings, encoding["attention_mask"])
        embeddings = self.projection(embeddings)
        embeddings = self.sparsifier(embeddings)
        if self.config.normalization_strategy == "l2":
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return BiEncoderEmbedding(embeddings, None, encoding)
