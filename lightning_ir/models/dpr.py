"""Configuration and model for DPR (Dense Passage Retriever) type models. Originally proposed in \
`Dense Passage Retrieval for Open-Domain Question Answering \
<https://arxiv.org/abs/2004.04906>`_. This model type is also known as a SentenceTransformer model:
`Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks \
<https://arxiv.org/abs/1908.10084>`_.
"""

from typing import Literal

import torch
from transformers import BatchEncoding

from ..bi_encoder import BiEncoderEmbedding, SingleVectorBiEncoderConfig, SingleVectorBiEncoderModel


class DprConfig(SingleVectorBiEncoderConfig):
    """Configuration class for a DPR model."""

    model_type = "lir-dpr"
    """Model type for a DPR model."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalize: bool = False,
        sparsification: Literal["relu", "relu_log"] | None = None,
        add_marker_tokens: bool = False,
        query_pooling_strategy: Literal["first", "mean", "max", "sum"] = "first",
        doc_pooling_strategy: Literal["first", "mean", "max", "sum"] = "first",
        embedding_dim: int | None = None,
        projection: Literal["linear", "linear_no_bias"] | None = "linear",
        **kwargs,
    ) -> None:
        """A DPR model encodes queries and documents separately. Before computing the similarity score, the
        contextualized token embeddings are aggregated to obtain a single embedding using a pooling strategy.
        Optionally, the pooled embeddings can be projected using a linear layer.

        :param query_length: Maximum query length, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length, defaults to 512
        :type doc_length: int, optional
        :param similarity_function: Similarity function to compute scores between query and document embeddings,
            defaults to "dot"
        :type similarity_function: Literal['cosine', 'dot'], optional
        :param sparsification: Whether and which sparsification function to apply, defaults to None
        :type sparsification: Literal['relu', 'relu_log'] | None, optional
        :param query_pooling_strategy: Whether and how to pool the query token embeddings, defaults to "first"
        :type query_pooling_strategy: Literal['first', 'mean', 'max', 'sum'], optional
        :param doc_pooling_strategy: Whether and how to pool document token embeddings, defaults to "first"
        :type doc_pooling_strategy: Literal['first', 'mean', 'max', 'sum'], optional

        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            normalize=normalize,
            sparsification=sparsification,
            add_marker_tokens=add_marker_tokens,
            query_pooling_strategy=query_pooling_strategy,
            doc_pooling_strategy=doc_pooling_strategy,
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

        :param config: Configuration for the DPR model
        :type config: SingleVectorBiEncoderConfig
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

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings and scoring mask.

        :param encoding: Tokenizer encodings for the text sequence
        :type encoding: BatchEncoding
        :param input_type: Type of input, either "query" or "doc"
        :type input_type: Literal["query", "doc"]
        :return: Embeddings and scoring mask
        :rtype: BiEncoderEmbedding
        """
        pooling_strategy = getattr(self.config, f"{input_type}_pooling_strategy")
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.pooling(embeddings, encoding["attention_mask"], pooling_strategy)
        embeddings = self.projection(embeddings)
        embeddings = self.sparsification(embeddings, self.config.sparsification)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return BiEncoderEmbedding(embeddings, None, encoding)
