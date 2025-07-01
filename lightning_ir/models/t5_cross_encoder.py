"""
Configuration and model implementation for T5 (Text-to-Text Transfer Transformer) type cross-encoder models.
Originally proposed in
`Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
<https://dl.acm.org/doi/10.5555/3455716.3455856>`_.
Two decoder strategies are supported:
- `mono`: The model predicts whether the document is relevant to the query.
- `rank`: The model predicts a relevance score for the document with respect to the query.
`RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses
<https://dl.acm.org/doi/10.1145/3539618.3592047>`_.
"""

from typing import Dict, Literal, Sequence, Type

import torch
from transformers import BatchEncoding

from ..cross_encoder.cross_encoder_config import CrossEncoderConfig
from ..cross_encoder.cross_encoder_model import CrossEncoderOutput
from ..cross_encoder.cross_encoder_tokenizer import CrossEncoderTokenizer
from .mono import MonoModel


class T5CrossEncoderConfig(CrossEncoderConfig):
    """Configuration class for a T5 cross-encoder model."""

    model_type = "encoder-decoder-cross-encoder"
    """Model type for a T5 cross-encoder model."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        decoder_strategy: Literal["mono", "rank"] = "mono",
        **kwargs,
    ) -> None:
        """A T5 cross-encoder model encodes queries and documents jointly. The contextualized embeddings are pooled
        into a single vector and fed to a linear layer which computes a final relevance score.

        :param query_length: Maximum query length, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length, defaults to 512
        :type doc_length: int, optional
        :param decoder_strategy: Strategy for the decoder, either "mono" for binary relevance or "rank" for ranking
            documents, defaults to "mono"
        :type decoder_strategy: Literal["mono", "rank"], optional
        """
        kwargs["pooling_strategy"] = "first"
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.decoder_strategy = decoder_strategy


class ScaleLinear(torch.nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586 # noqa
        input = input * (input.shape[-1] ** -0.5)
        return super().forward(input)


class T5CrossEncoderModel(MonoModel):
    """T5 cross-encoder model. See :class:`T5CrossEncoderConfig` for configuration options."""

    config_class = T5CrossEncoderConfig
    """Configuration class for T5 cross-encoder models."""

    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "linear.weight"]
    """Keys of the weights that are tied between the encoder and decoder."""

    def __init__(self, config: T5CrossEncoderConfig, *args, **kwargs):
        """Initializes a T5 cross-encoder model given a :class:`T5CrossEncoderConfig`.

        :param config: Configuration for the T5 cross-encoder model
        :type config: T5CrossEncoderConfig
        """
        super().__init__(config, *args, **kwargs)
        self.config: T5CrossEncoderConfig
        if self.config.decoder_strategy == "mono":
            self.linear = ScaleLinear(config.hidden_size, 2, bias=config.linear_bias)
        else:
            self.linear = ScaleLinear(config.hidden_size, 1, bias=config.linear_bias)

    # TODO tieing of weights does not work when setting linear to only use slice of lm head for efficiency
    # def get_output_embeddings(self):
    #     shared = self.shared
    #     if self.config.decoder_strategy == "mono":
    #         self.linear.weight.data = shared.weight.data[[1176, 6136]]
    #     elif self.config.decoder_strategy == "rank":
    #         self.linear.weight.data = shared.weight.data[[32089]]
    #     else:
    #         raise ValueError("Unknown decoder strategy")
    #     return shared

    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        """Computes contextualized embeddings for the joint query-document input sequence and computes a relevance
        score.

        :param encoding: Tokenizer encoding for the joint query-document input sequence
        :type encoding: BatchEncoding
        :return: Output of the model
        :rtype: CrossEncoderOutput
        """
        decoder_input_ids = torch.zeros(
            (encoding["input_ids"].shape[0], 1), device=encoding["input_ids"].device, dtype=torch.long
        )
        encoding["decoder_input_ids"] = decoder_input_ids
        output = super().forward(encoding)
        if output.scores is None:
            raise ValueError("Scores are None")
        if self.config.decoder_strategy == "mono":
            scores = output.scores.view(-1, 2)
            scores = torch.nn.functional.log_softmax(scores, dim=-1)[:, 0]
            output.scores = scores.view(-1)
        return output


class T5CrossEncoderTokenizer(CrossEncoderTokenizer):
    """Tokenizer for T5 cross-encoder models. It formats the input text according to the decoder strategy."""

    config_class: Type[T5CrossEncoderConfig] = T5CrossEncoderConfig
    """Configuration class for T5 cross-encoder tokenizers."""

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        decoder_strategy: Literal["mono", "rank"] = "mono",
        **kwargs,
    ):
        """Initializes a T5 cross-encoder tokenizer.

        :param query_length: Maximum query length, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length, defaults to 512
        :type doc_length: int, optional
        :param decoder_strategy: Decoder strategy, defaults to "mono"
        :type decoder_strategy: Literal["mono", "rank"], optional
        """
        super().__init__(
            *args, query_length=query_length, doc_length=doc_length, decoder_strategy=decoder_strategy, **kwargs
        )
        self.decoder_strategy = decoder_strategy

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        num_docs: Sequence[int] | int | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        """Tokenizes queries and documents into a single sequence of tokens.

        :param queries: Query or list of queries to tokenize, defaults to None
        :type queries: str | Sequence[str] | None, optional
        :param docs: Document or list of documents to tokenize, defaults to None
        :type docs: str | Sequence[str] | None, optional
        :param num_docs: Number of documents per query, defaults to None
        :type num_docs: Sequence[int] | int | None, optional
        :return: Tokenized queries and documents
        :rtype: Dict[str, BatchEncoding]
        """
        expanded_queries, docs = self._preprocess(queries, docs, num_docs)
        if self.decoder_strategy == "mono":
            pattern = "Query: {query} Document: {doc} Relevant:"
        elif self.decoder_strategy == "rank":
            pattern = "Query: {query} Document: {doc}"
        else:
            raise ValueError(f"Unknown decoder strategy: {self.decoder_strategy}")
        input_texts = [pattern.format(query=query, doc=doc) for query, doc in zip(expanded_queries, docs)]

        return_tensors = kwargs.get("return_tensors", None)
        if return_tensors is not None:
            kwargs["pad_to_multiple_of"] = 8
        return {"encoding": self(input_texts, **kwargs)}
