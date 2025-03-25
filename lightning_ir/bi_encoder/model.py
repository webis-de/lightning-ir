"""
Model module for bi-encoder models.

This module defines the model class used to implement bi-encoder models.
"""

import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from string import punctuation
from typing import Callable, Iterable, Literal, Sequence, Tuple, Type, overload

import torch
from transformers import BatchEncoding

from ..base import LightningIRModel, LightningIROutput
from ..base.model import batch_encoding_wrapper
from ..modeling_utils.mlm_head import (
    MODEL_TYPE_TO_HEAD_NAME,
    MODEL_TYPE_TO_LM_HEAD,
    MODEL_TYPE_TO_OUTPUT_EMBEDDINGS,
    MODEL_TYPE_TO_TIED_WEIGHTS_KEYS,
)
from . import BiEncoderConfig


@dataclass
class BiEncoderEmbedding:
    """Dataclass containing embeddings and scoring mask for bi-encoder models."""

    embeddings: torch.Tensor
    """Embedding tensor generated by a bi-encoder model of shape [batch_size x seq_len x hidden_size]. The sequence
    length varies depending on the pooling strategy and the hidden size varies depending on the projection settings."""
    scoring_mask: torch.Tensor
    """Mask tensor designating which vectors should be ignored during scoring."""
    encoding: BatchEncoding | None
    """Tokenizer encodings used to generate the embeddings."""

    @overload
    def to(self, device: torch.device, /) -> "BiEncoderEmbedding": ...

    @overload
    def to(self, other: "BiEncoderEmbedding", /) -> "BiEncoderEmbedding": ...

    def to(self, device) -> "BiEncoderEmbedding":
        """Moves the embeddings and scoring mask to the specified device.

        :param device: Device to move the embeddings to or another BiEncoderEmbedding object to move to the same device
        :type device: torch.device | BiEncoderEmbedding
        :return: Self
        :rtype: BiEncoderEmbedding
        """
        if isinstance(device, BiEncoderEmbedding):
            device = device.device
        self.embeddings = self.embeddings.to(device)
        self.scoring_mask = self.scoring_mask.to(device)
        self.encoding = self.encoding.to(device)
        return self

    @property
    def device(self) -> torch.device:
        """Returns the device of the embeddings.

        :raises ValueError: If the embeddings and scoring_mask are not on the same device
        :return: The device of the embeddings
        :rtype: torch.device
        """
        if self.embeddings.device != self.scoring_mask.device:
            raise ValueError("Embeddings and scoring_mask must be on the same device")
        return self.embeddings.device

    def items(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """Iterates over the embeddings attributes and their values like `dict.items()`.

        :yield: Tuple of attribute name and its value
        :rtype: Iterator[Iterable[Tuple[str, torch.Tensor]]]
        """
        for field in self.__dataclass_fields__:
            yield field, getattr(self, field)


@dataclass
class BiEncoderOutput(LightningIROutput):
    """Dataclass containing the output of a bi-encoder model."""

    query_embeddings: BiEncoderEmbedding | None = None
    """Query embeddings and scoring_mask generated by the model."""
    doc_embeddings: BiEncoderEmbedding | None = None
    """Document embeddings and scoring_mask generated by the model."""


class BiEncoderModel(LightningIRModel):

    config_class: Type[BiEncoderConfig] = BiEncoderConfig
    """Configuration class for the bi-encoder model."""

    def __init__(self, config: BiEncoderConfig, *args, **kwargs) -> None:
        """A bi-encoder model that encodes queries and documents separately and computes a relevance score between them
        using a :class:`.ScoringFunction`. See :class:`.BiEncoderConfig` for configuration options.

        :param config: Configuration for the bi-encoder model
        :type config: BiEncoderConfig
        :raises ValueError: If a projection is used but the hidden size of the backbone encoder and embedding dim of the
            bi-encoder model do not match
        """
        super().__init__(config, *args, **kwargs)
        self.config: BiEncoderConfig
        self.scoring_function = ScoringFunction(self.config)
        projection: torch.nn.Module = torch.nn.Identity()
        if self.config.projection is not None:
            if "linear" in self.config.projection:
                projection = torch.nn.Linear(
                    self.config.hidden_size,
                    self.config.embedding_dim,
                    bias="no_bias" not in self.config.projection,
                )
            elif self.config.projection == "mlm":
                layer_cls = MODEL_TYPE_TO_LM_HEAD[config.backbone_model_type or config.model_type]
                projection = layer_cls(config)
                tied_weight_keys = getattr(self, "_tied_weights_keys", []) or []
                projection_layers = (
                    ("projection",) if self.config.tie_projection else ("query_projection", "doc_projection")
                )
                tied_weight_keys = tied_weight_keys + [
                    f"{projection_layer}.{key}"
                    for key in MODEL_TYPE_TO_TIED_WEIGHTS_KEYS[config.backbone_model_type or config.model_type]
                    for projection_layer in projection_layers
                ]
                setattr(self, "_tied_weights_keys", tied_weight_keys)
            else:
                raise ValueError(f"Unknown projection {self.config.projection}")
        else:
            if self.config.embedding_dim != self.config.hidden_size:
                warnings.warn(
                    "No projection is used but embedding_dim != hidden_size. "
                    "The output embeddings will not have embedding_size dimensions."
                )

        if config.tie_projection:
            self.projection = projection
        else:
            self.query_projection = deepcopy(projection)
            self.doc_projection = deepcopy(projection)

        self.query_mask_scoring_input_ids: torch.Tensor | None = None
        self.doc_mask_scoring_input_ids: torch.Tensor | None = None
        self._add_mask_scoring_input_ids()

    @classmethod
    def _load_pretrained_model(
        cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
    ):
        if model.config.projection == "mlm":
            head_name = MODEL_TYPE_TO_HEAD_NAME[model.config.backbone_model_type or model.config.model_type]
            projection_layers = (
                ("projection",) if model.config.tie_projection else ("query_projection", "doc_projection")
            )
            for loaded_key in list(loaded_keys):
                if loaded_key.startswith(head_name):
                    for projection_layer in projection_layers:
                        new_key = loaded_key.replace(head_name, projection_layer)
                        loaded_keys.append(new_key)
                        state_dict[new_key] = state_dict[loaded_key].clone()

        return super()._load_pretrained_model(
            model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
        )

    def get_output_embeddings(self) -> torch.nn.Module | None:
        """Returns the output embeddings of the model for tieing the input and output embeddings. Returns None if no
        MLM head is used for projection.

        :return: Output embeddings of the model
        :rtype: torch.nn.Module | None
        """

        class _TiedOutputEmbeddingsContainer(torch.nn.Module):
            """This is a hack to tie the output embeddings of multiple layers. HF only supports tieing the output of a
            single layer at the moment. This hack will lead to errors if the input embedding dimensionality is changed,
            e.g., a new token is added to the vocabulary."""

            def __init__(self, output_embeddings: Sequence[torch.nn.Linear]):
                super().__init__()
                self.output_embeddings = output_embeddings
                self.weight = output_embeddings[0].weight

            def __setattr__(self, name: str, value: torch.Tensor | torch.nn.Module) -> None:
                if name == "weight":
                    for output_embedding in self.output_embeddings:
                        output_embedding.weight = value
                super().__setattr__(name, value)

        if self.config.projection == "mlm":
            if self.config.tie_projection:
                return self.projection
            else:
                module_names = MODEL_TYPE_TO_OUTPUT_EMBEDDINGS[
                    self.config.backbone_model_type or self.config.model_type
                ]
                query_output = self.query_projection
                doc_output = self.doc_projection
                for module_name in module_names.split("."):
                    query_output = getattr(query_output, module_name)
                    doc_output = getattr(doc_output, module_name)
                container = _TiedOutputEmbeddingsContainer([query_output, doc_output])
                return container
        return None

    def set_output_embeddings(self, new_embeddings: torch.nn.Module) -> None:
        if self.config.projection == "mlm":
            raise NotImplementedError("Setting output embeddings is not supported for models with MLM projection.")
            module_names = MODEL_TYPE_TO_OUTPUT_EMBEDDINGS[self.config.backbone_model_type or self.config.model_type]
            module = self
            for module_name in module_names.split(".")[:-1]:
                module = getattr(module, module_name)
            setattr(module, module_names.split(".")[-1], new_embeddings)
            setattr(module, "bias", new_embeddings.bias)

    def _add_mask_scoring_input_ids(self) -> None:
        """Adds the mask scoring input ids to the model if they are specified in the configuration."""
        for sequence in ("query", "doc"):
            mask_scoring_tokens = getattr(self.config, f"{sequence}_mask_scoring_tokens")
            if mask_scoring_tokens is None:
                continue
            if mask_scoring_tokens == "punctuation":
                mask_scoring_tokens = list(punctuation)
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
            except OSError:
                raise ValueError("Can't use token scoring masking if the checkpoint does not have a tokenizer.")
            mask_scoring_input_ids = []
            for token in mask_scoring_tokens:
                if token not in tokenizer.vocab:
                    raise ValueError(f"Token {token} not in tokenizer vocab")
                mask_scoring_input_ids.append(tokenizer.vocab[token])
            setattr(
                self,
                f"{sequence}_mask_scoring_input_ids",
                torch.tensor(mask_scoring_input_ids, dtype=torch.long),
            )

    def forward(
        self,
        query_encoding: BatchEncoding | None,
        doc_encoding: BatchEncoding | None,
        num_docs: Sequence[int] | int | None = None,
    ) -> BiEncoderOutput:
        """Embeds queries and/or documents and computes relevance scores between them if both are provided.

        :param query_encoding: Tokenizer encodings for the queries
        :type query_encoding: BatchEncoding | None
        :param doc_encoding: Tokenizer encodings for the documents
        :type doc_encoding: BatchEncoding | None
        :param num_docs: Specifies how many documents are passed per query. If a sequence of integers, `len(num_doc)`
            should be equal to the number of queries and `sum(num_docs)` equal to the number of documents, i.e., the
            sequence contains one value per query specifying the number of documents for that query. If an integer,
            assumes an equal number of documents per query. If None, tries to infer the number of documents by dividing
            the number of documents by the number of queries, defaults to None
        :type num_docs: Sequence[int] | int | None, optional
        :return: Output of the model
        :rtype: BiEncoderOutput
        """
        query_embeddings = None
        if query_encoding is not None:
            query_embeddings = self.encode_query(query_encoding)
        doc_embeddings = None
        if doc_encoding is not None:
            doc_embeddings = self.encode_doc(doc_encoding)
        scores = None
        if doc_embeddings is not None and query_embeddings is not None:
            scores = self.score(query_embeddings, doc_embeddings, num_docs)
        return BiEncoderOutput(scores=scores, query_embeddings=query_embeddings, doc_embeddings=doc_embeddings)

    def encode_query(self, encoding: BatchEncoding) -> BiEncoderEmbedding:
        """Encodes tokenized queries.

        :param encoding: Tokenizer encodings for the queries
        :type encoding: BatchEncoding
        :return: Query embeddings and scoring mask
        :rtype: BiEncoderEmbedding
        """
        return self.encode(encoding=encoding, input_type="query")

    def encode_doc(self, encoding: BatchEncoding) -> BiEncoderEmbedding:
        """Encodes tokenized documents.

        :param encoding: Tokenizer encodings for the documents
        :type encoding: BatchEncoding
        :return: Query embeddings and scoring mask
        :rtype: BiEncoderEmbedding
        """
        return self.encode(encoding=encoding, input_type="doc")

    @batch_encoding_wrapper
    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings ad scoring mask.

        :param encoding: Tokenizer encodings for the text sequence
        :type encoding: BatchEncoding
        :param input_type: Type of input, either "query" or "doc"
        :type input_type: Literal["query", "doc"]
        :return: Embeddings and scoring mask
        :rtype: BiEncoderEmbedding
        """
        expansion = getattr(self.config, f"{input_type}_expansion")
        pooling_strategy = getattr(self.config, f"{input_type}_pooling_strategy")
        projection = self.projection if self.config.tie_projection else getattr(self, f"{input_type}_projection")
        mask_scoring_input_ids = getattr(self, f"{input_type}_mask_scoring_input_ids")

        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = projection(embeddings)
        embeddings = self._sparsification(embeddings, self.config.sparsification)
        embeddings = self._pooling(embeddings, encoding["attention_mask"], pooling_strategy)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self.scoring_mask(encoding, expansion, pooling_strategy, mask_scoring_input_ids)
        return BiEncoderEmbedding(embeddings, scoring_mask, encoding)

    def scoring_mask(
        self,
        encoding: BatchEncoding,
        expansion: bool = False,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
        mask_scoring_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes a scoring for batched tokenized text sequences which is used in the scoring function to mask out
        vectors during scoring.

        :param encoding: Tokenizer encodings for the text sequence
        :type encoding: BatchEncoding
        :param expansion: Whether or not mask expansion was applied to the tokenized sequence, defaults to False
        :type expansion: bool, optional
        :param pooling_strategy: Which pooling strategy is pool the embeddings, defaults to None
        :type pooling_strategy: Literal['first', 'mean', 'max', 'sum'] | None, optional
        :param mask_scoring_input_ids: Sequence of token_ids which should be masked during scoring, defaults to None
        :type mask_scoring_input_ids: torch.Tensor | None, optional
        :return: Scoring mask
        :rtype: torch.Tensor
        """
        device = encoding["input_ids"].device
        input_ids: torch.Tensor = encoding["input_ids"]
        attention_mask: torch.Tensor = encoding["attention_mask"]
        shape = input_ids.shape
        if pooling_strategy is not None:
            return torch.ones((shape[0], 1), dtype=torch.bool, device=device)
        scoring_mask = attention_mask
        if expansion or scoring_mask is None:
            scoring_mask = torch.ones(shape, dtype=torch.bool, device=device)
        scoring_mask = scoring_mask.bool()
        if mask_scoring_input_ids is not None:
            ignore_mask = input_ids[..., None].eq(mask_scoring_input_ids.to(device)).any(-1)
            scoring_mask = scoring_mask & ~ignore_mask
        return scoring_mask

    def score(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Compute relevance scores between queries and documents.

        :param query_embeddings: Embeddings and scoring mask for the queries
        :type query_embeddings: BiEncoderEmbedding
        :param doc_embeddings: Embeddings and scoring mask for the documents
        :type doc_embeddings: BiEncoderEmbedding
        :param num_docs: Specifies how many documents are passed per query. If a sequence of integers, `len(num_doc)`
            should be equal to the number of queries and `sum(num_docs)` equal to the number of documents, i.e., the
            sequence contains one value per query specifying the number of documents for that query. If an integer,
            assumes an equal number of documents per query. If None, tries to infer the number of documents by dividing
            the number of documents by the number of queries, defaults to None
        :type num_docs: Sequence[int] | int | None, optional
        :return: Relevance scores
        :rtype: torch.Tensor
        """
        scores = self.scoring_function(query_embeddings, doc_embeddings, num_docs=num_docs)
        return scores


def _batch_scoring(
    similarity_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Helper function to batch similarity functions to avoid memory issues with large batch sizes or high numbers
    of documents per query."""
    BATCH_SIZE = 16384

    @wraps(similarity_function)
    def batch_similarity_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= BATCH_SIZE:
            return similarity_function(x, y)
        out = torch.zeros(x.shape[0], x.shape[1], y.shape[2], device=x.device, dtype=x.dtype)
        for i in range(0, x.shape[0], BATCH_SIZE):
            out[i : i + BATCH_SIZE] = similarity_function(x[i : i + BATCH_SIZE], y[i : i + BATCH_SIZE])
        return out

    return batch_similarity_function


class ScoringFunction(torch.nn.Module):
    def __init__(self, config: BiEncoderConfig) -> None:
        """Scoring function for bi-encoder models. Computes similarity scores between query and document embeddings. For
        multi-vector models, the scores are aggregated to a single score per query-document pair.

        :param config: Configuration for the bi-encoder model
        :type config: BiEncoderConfig
        :raises ValueError: If the similarity function is not supported
        """
        super().__init__()
        self.config = config
        if self.config.similarity_function == "cosine":
            self.similarity_function = self._cosine_similarity
        elif self.config.similarity_function == "l2":
            self.similarity_function = self._l2_similarity
        elif self.config.similarity_function == "dot":
            self.similarity_function = self._dot_similarity
        else:
            raise ValueError(f"Unknown similarity function {self.config.similarity_function}")
        self.query_aggregation_function = self.config.query_aggregation_function

    def compute_similarity(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Computes the similarity score between all query and document embedding vector pairs."""
        # TODO compute similarity only for non-masked values
        num_docs_t = self._parse_num_docs(
            query_embeddings.embeddings.shape[0], doc_embeddings.embeddings.shape[0], num_docs, query_embeddings.device
        )
        query_emb = query_embeddings.embeddings.repeat_interleave(num_docs_t, dim=0).unsqueeze(2)
        doc_emb = doc_embeddings.embeddings.unsqueeze(1)
        similarity = self.similarity_function(query_emb, doc_emb)
        return similarity

    @staticmethod
    @_batch_scoring
    @torch.autocast(device_type="cuda", enabled=False)
    def _cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)

    @staticmethod
    @_batch_scoring
    @torch.autocast(device_type="cuda", enabled=False)
    def _l2_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -1 * torch.cdist(x, y).squeeze(-2)

    @staticmethod
    @_batch_scoring
    @torch.autocast(device_type="cuda", enabled=False)
    def _dot_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.transpose(-1, -2)).squeeze(-2)

    def _parse_num_docs(
        self, query_shape: int, doc_shape: int, num_docs: int | Sequence[int] | None, device: torch.device | None = None
    ) -> torch.Tensor:
        """Helper function to parse the number of documents per query."""
        if isinstance(num_docs, int):
            num_docs = [num_docs] * query_shape
        if isinstance(num_docs, list):
            if sum(num_docs) != doc_shape or len(num_docs) != query_shape:
                raise ValueError("Num docs does not match doc embeddings")
        if num_docs is None:
            if doc_shape % query_shape != 0:
                raise ValueError("Docs are not evenly distributed in _batch, but no num_docs provided")
            num_docs = [doc_shape // query_shape] * query_shape
        return torch.tensor(num_docs, device=device)

    def _expand_mask(self, shape: torch.Size, mask: torch.Tensor, dim: int) -> torch.Tensor:
        if mask.ndim == len(shape):
            return mask
        if mask.ndim > len(shape):
            raise ValueError("Mask has too many dimensions")
        fill_values = len(shape) - mask.ndim + 1
        new_shape = [*mask.shape[:-1]] + [1] * fill_values
        new_shape[dim] = mask.shape[-1]
        return mask.view(*new_shape)

    def _aggregate(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        query_aggregation_function: Literal["max", "sum", "mean", "harmonic_mean"] | None,
        dim: int,
    ) -> torch.Tensor:
        """Helper function to aggregate similarity scores over query and document embeddings."""
        mask = self._expand_mask(scores.shape, mask, dim)
        if query_aggregation_function is None:
            return scores
        if query_aggregation_function == "max":
            scores = scores.masked_fill(~mask, float("-inf"))
            return scores.amax(dim, keepdim=True)
        if query_aggregation_function == "sum":
            scores = scores.masked_fill(~mask, 0)
            return scores.sum(dim, keepdim=True)
        num_non_masked = mask.sum(dim, keepdim=True)
        if query_aggregation_function == "mean":
            return torch.where(num_non_masked == 0, 0, scores.sum(dim, keepdim=True) / num_non_masked)
        if query_aggregation_function == "harmonic_mean":
            return torch.where(
                num_non_masked == 0,
                0,
                num_non_masked / (1 / scores).sum(dim, keepdim=True),
            )
        raise ValueError(f"Unknown aggregation {query_aggregation_function}")

    def aggregate_similarity(
        self,
        similarity: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: int | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Aggregates the matrix of query-document similarities into a single score based on the configured aggregation
        strategy.

        :param similarity: Query-document similarity matrix
        :type similarity: torch.Tensor
        :param query_scoring_mask: Which query vectors should be masked out during scoring
        :type query_scoring_mask: torch.Tensor
        :param doc_scoring_mask: Which doucment vectors should be masked out during scoring
        :type doc_scoring_mask: torch.Tensor
        :return: Aggregated similarity scores
        :rtype: torch.Tensor
        """
        num_docs_t = self._parse_num_docs(
            query_scoring_mask.shape[0], doc_scoring_mask.shape[0], num_docs, similarity.device
        )
        scores = self._aggregate(similarity, doc_scoring_mask, "max", -1)
        scores = self._aggregate(
            scores, query_scoring_mask.repeat_interleave(num_docs_t, dim=0), self.query_aggregation_function, -2
        )
        return scores.view(scores.shape[0])

    def forward(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Compute relevance scores between query and document embeddings.

        :param query_embeddings: Embeddings and scoring mask for the queries
        :type query_embeddings: BiEncoderEmbedding
        :param doc_embeddings: Embeddings and scoring mask for the documents
        :type doc_embeddings: BiEncoderEmbedding
        :param num_docs: Specifies how many documents are passed per query. If a sequence of integers, `len(num_doc)`
            should be equal to the number of queries and `sum(num_docs)` equal to the number of documents, i.e., the
            sequence contains one value per query specifying the number of documents for that query. If an integer,
            assumes an equal number of documents per query. If None, tries to infer the number of documents by dividing
            the number of documents by the number of queries, defaults to None
        :type num_docs: Sequence[int] | int | None, optional
        :return: Relevance scores
        :rtype: torch.Tensor
        """
        similarity = self.compute_similarity(query_embeddings, doc_embeddings, num_docs)
        return self.aggregate_similarity(similarity, query_embeddings.scoring_mask, doc_embeddings.scoring_mask)
