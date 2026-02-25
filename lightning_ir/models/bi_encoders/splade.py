"""Configuration and model for SPLADE (SParse Lexical AnD Expansion) type models.

SPLADE (SParse Lexical AnD Expansion) is a efficient retrieval model that bridges the gap between traditional
keyword search and deep neural understanding. It uses a language model to analyze a document and assign importance
scores to words across the entire vocabulary, expanding the text with highly relevant terms that were not originally
present. This process creates a sparse, high-dimensional vector that can be stored and searched using inverted indices
just like traditional search engines.

Originally proposed in
`SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking
<https://dl.acm.org/doi/abs/10.1145/3404835.3463098>`_.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import torch
from transformers import BatchEncoding

from ...bi_encoder import (
    BiEncoderEmbedding,
    BiEncoderTokenizer,
    SingleVectorBiEncoderConfig,
    SingleVectorBiEncoderModel,
)
from ...modeling_utils.embedding_post_processing import Pooler, Sparsifier
from ...modeling_utils.lm_head import MODEL_TYPE_TO_LM_HEAD, MODEL_TYPE_TO_STATE_DICT_KEY_MAPPING


class SpladeConfig(SingleVectorBiEncoderConfig):
    """Configuration class for a SPLADE model."""

    model_type = "splade"
    """Model type for a SPLADE model."""

    def __init__(
        self,
        query_length: int | None = 32,
        doc_length: int | None = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        sparsification_strategy: Literal["relu", "relu_log", "relu_2xlog"] | None = "relu_log",
        pooling_strategy: Literal["first", "mean", "max", "sum"] = "max",
        query_weighting: Literal["contextualized", "static"] | None = "contextualized",
        query_expansion: bool = True,
        doc_weighting: Literal["contextualized", "static"] | None = "contextualized",
        doc_expansion: bool = True,
        **kwargs,
    ) -> None:
        """A SPLADE model encodes queries and documents separately. Before computing the similarity score, the
        contextualized token embeddings are projected into a logit distribution over the vocabulary using a pre-trained
        masked language model (MLM) head. The logit distribution is then sparsified and aggregated to obtain a single
        embedding for the query and document.

        Args:
            query_length (int | None): Maximum number of tokens per query. If None does not truncate. Defaults to 32.
            doc_length (int | None): Maximum number of tokens per document. If None does not truncate. Defaults to 512.
            similarity_function (Literal["cosine", "dot"]): Similarity function to compute scores between query and
                document embeddings. Defaults to "dot".
            sparsification_strategy (Literal['relu', 'relu_log', 'relu_2xlog'] | None): Whether and which sparsification
                function to apply. Defaults to None.
            pooling_strategy (Literal["first", "mean", "max", "sum"]): Pooling strategy for query embeddings.
                Defaults to "max".
            query_weighting (Literal["contextualized", "static"] | None): Whether to reweight query embeddings.
                Defaults to "contextualized".
            query_expansion (bool): Whether to allow query expansion. Defaults to True.
            doc_weighting (Literal["contextualized", "static"] | None): Whether to reweight document embeddings.
                Defaults to "contextualized".
            doc_expansion (bool): Whether to allow document expansion. Defaults to True.
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            sparsification_strategy=sparsification_strategy,
            pooling_strategy=pooling_strategy,
            **kwargs,
        )
        if query_expansion and not query_weighting:
            raise ValueError("If query_expansion is True, query_weighting must also be True.")
        if doc_expansion and not doc_weighting:
            raise ValueError("If doc_expansion is True, doc_weighting must also be True.")
        self.query_weighting = query_weighting
        self.query_expansion = query_expansion
        self.doc_weighting = doc_weighting
        self.doc_expansion = doc_expansion

    @property
    def embedding_dim(self) -> int:
        vocab_size = getattr(self, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("Unable to determine embedding dimension.")
        return vocab_size

    @embedding_dim.setter
    def embedding_dim(self, value: int) -> None:
        pass


class SpladeModel(SingleVectorBiEncoderModel):
    """Sparse lexical SPLADE model. See :class:`SpladeConfig` for configuration options."""

    config_class = SpladeConfig
    """Configuration class for a SPLADE model."""

    def __init__(self, config: SingleVectorBiEncoderConfig, *args, **kwargs) -> None:
        """Initializes a SPLADE model given a :class:`SpladeConfig`.

        Args:
            config (SingleVectorBiEncoderConfig): Configuration for the SPLADE model.
        """
        super().__init__(config, *args, **kwargs)
        # grab language modeling head based on backbone model type
        layer_cls = MODEL_TYPE_TO_LM_HEAD[config.backbone_model_type or config.model_type]
        self.projection = layer_cls(config)
        existing_tied = getattr(self, "_tied_weights_keys", None) or {}
        if isinstance(existing_tied, dict):
            # transformers v5: _tied_weights_keys is a dict {tied_key: source_key}
            tied_weight_keys: dict | list = {**existing_tied, "projection.decoder.weight": "embeddings.word_embeddings.weight", }
        else:
            # transformers v4: _tied_weights_keys is a list of tied key names
            tied_weight_keys = list(existing_tied) + ["projection.decoder.weight"]
        self._tied_weights_keys = tied_weight_keys
        self.query_weights = None
        if config.query_weighting == "static":
            self.query_weights = torch.nn.Embedding(config.vocab_size, 1)
        self.doc_weights = None
        if config.doc_weighting == "static":
            self.doc_weights = torch.nn.Embedding(config.vocab_size, 1)

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
        weighting = getattr(self.config, f"{input_type}_weighting")
        expansion = getattr(self.config, f"{input_type}_expansion")
        token_mask = None
        if weighting is None or weighting == "static" or not expansion:
            token_mask = torch.zeros(
                encoding["input_ids"].shape[0],
                self.config.embedding_dim,
                device=encoding["input_ids"].device,
                dtype=torch.float32,
            )
            if weighting == "static":
                weights = getattr(self, f"{input_type}_weights")(encoding["input_ids"]).squeeze(-1)
            else:
                weights = encoding["attention_mask"].to(token_mask)
            weights = weights.masked_fill(~(encoding["attention_mask"].bool()), 0.0)
            token_mask = token_mask.scatter(1, encoding["input_ids"], weights)
            if weighting is None or weighting == "static":
                # inference free
                return BiEncoderEmbedding(token_mask[:, None], None, encoding)

        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.projection(embeddings)
        embeddings = self.sparsifier(embeddings)
        embeddings = self.pooler(embeddings, encoding["attention_mask"])
        if token_mask is not None:
            embeddings = embeddings * token_mask[:, None]
        return BiEncoderEmbedding(embeddings, None, encoding)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path, *args, **kwargs) -> Self:
        """Loads a pretrained model and handles mapping the MLM head weights to the projection head weights. Wraps
        the transformers.PreTrainedModel.from_pretrained_ method to return a derived LightningIRModel.
        See :class:`LightningIRModelClassFactory` for more details.

.. _transformers.PreTrainedModel.from_pretrained: \
    https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

        .. ::doctest
        .. highlight:: python
        .. code-block:: python

            >>> # Loading using model class and backbone checkpoint
            >>> type(CrossEncoderModel.from_pretrained("bert-base-uncased"))
            <class 'lightning_ir.base.class_factory.CrossEncoderBertModel'>
            >>> # Loading using base class and backbone checkpoint
            >>> type(LightningIRModel.from_pretrained("bert-base-uncased", config=CrossEncoderConfig()))
            <class 'lightning_ir.base.class_factory.CrossEncoderBertModel'>

        Args:
            model_name_or_path (str | Path): Name or path of the pretrained model.
        Returns:
            Self: A derived LightningIRModel consisting of a backbone model and a LightningIRModel mixin.
        Raises:
            ValueError: If called on the abstract class :class:`SpladeModel` and no config is passed.
        """
        key_mapping = kwargs.pop("key_mapping", {})
        config = cls.config_class
        # map mlm projection keys
        model_type = config.backbone_model_type or config.model_type
        if model_type in MODEL_TYPE_TO_STATE_DICT_KEY_MAPPING:
            key_mapping.update(MODEL_TYPE_TO_STATE_DICT_KEY_MAPPING[model_type])
        if not key_mapping:
            warnings.warn(
                f"No mlm key mappings for model_type {model_type} were provided. "
                "The pre-trained mlm weights will not be loaded correctly.",
                stacklevel=2,
            )
        model = super().from_pretrained(model_name_or_path, *args, key_mapping=key_mapping, **kwargs)
        return model

    def set_output_embeddings(self, new_embeddings: torch.nn.Module) -> None:
        if self.config.projection == "mlm":
            raise NotImplementedError("Setting output embeddings is not supported for models with MLM projection.")
            # TODO fix this (not super important, only necessary when additional tokens are added to the model)
            # module_names = MODEL_TYPE_TO_OUTPUT_EMBEDDINGS[self.config.backbone_model_type or self.config.model_type]
            # module = self
            # for module_name in module_names.split(".")[:-1]:
            #     module = getattr(module, module_name)
            # setattr(module, module_names.split(".")[-1], new_embeddings)
            # setattr(module, "bias", new_embeddings.bias)

    def get_output_embeddings(self) -> torch.nn.Module | None:
        """Returns the output embeddings of the model for tieing the input and output embeddings. Returns None if no
        MLM head is used for projection.

        Returns:
            torch.nn.Module | None: Output embeddings of the model.
        """
        return self.projection.decoder


class SpladeTokenizer(BiEncoderTokenizer):
    """Tokenizer class for SPLADE models."""

    config_class = SpladeConfig
    """Configuration class for a SPLADE model."""

    def __init__(
        self,
        *args,
        query_length: int | None = 32,
        doc_length: int | None = 512,
        add_marker_tokens: bool = False,
        query_weighting: Literal["contextualized", "static"] = "contextualized",
        doc_weighting: Literal["contextualized", "static"] = "contextualized",
        **kwargs,
    ):
        super().__init__(
            *args, query_length=query_length, doc_length=doc_length, add_marker_tokens=add_marker_tokens, **kwargs
        )
        """Initializes a SPLADE model's tokenizer. Encodes queries and documents separately. Optionally adds
        marker tokens to encoded input sequences.

        Args:
            query_length (int | None): Maximum number of tokens per query. If None does not truncate. Defaults to 32.
            doc_length (int | None): Maximum number of tokens per document. If None does not truncate. Defaults to 512.
            add_marker_tokens (bool): Whether to add extra marker tokens [Q] / [D] to queries / documents.
                Defaults to False.
            query_expansion (bool): Whether to expand queries with mask tokens. Defaults to False.
            attend_to_query_expanded_tokens (bool): Whether to allow query tokens to attend to mask expanded query
                tokens. Defaults to False.
            doc_expansion (bool): Whether to expand documents with mask tokens. Defaults to False.
            attend_to_doc_expanded_tokens (bool): Whether to allow document tokens to attend to mask expanded document
                tokens. Defaults to False.
            query_weighting (Literal["contextualized", "static"]): Whether to apply weighting to query tokens.
                Defaults to "contextualized".
            doc_weighting (Literal["contextualized", "static"]): Whether to apply weighting to document tokens.
                Defaults to "contextualized".
        Raises:
            ValueError: If `add_marker_tokens` is True and a non-supported tokenizer is used.
        """
        super().__init__(
            *args,
            query_length=query_length,
            doc_length=doc_length,
            add_marker_tokens=add_marker_tokens,
            query_weighting=query_weighting,
            doc_weighting=doc_weighting,
            **kwargs,
        )
        self.query_weighting = query_weighting
        self.doc_weighting = doc_weighting

    def tokenize_input_sequence(
        self, text: Sequence[str] | str, input_type: Literal["query", "doc"], *args, **kwargs
    ) -> BatchEncoding:
        """Tokenizes an input sequence. This method is used to tokenize both queries and documents.

        Args:
            text (Sequence[str] | str): Input text to tokenize.
            input_type (Literal["query", "doc"]): type of input, either "query" or "doc".
        Returns:
            BatchEncoding: Tokenized input sequences.
        """
        post_processer = getattr(self, f"{input_type}_post_processor")
        kwargs["max_length"] = getattr(self, f"{input_type}_length")
        weighting = getattr(self, f"{input_type}_weighting")
        if weighting is None or weighting == "static":
            kwargs["add_special_tokens"] = False
        if "padding" not in kwargs:
            kwargs["truncation"] = True
        return self._encode(text, *args, post_processor=post_processer, **kwargs)
