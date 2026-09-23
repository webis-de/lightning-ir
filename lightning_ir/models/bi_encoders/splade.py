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
from ...modeling_utils.lm_head import (
    MODEL_TYPE_TO_INPUT_EMBEDDINGS_KEY,
    MODEL_TYPE_TO_LM_HEAD,
    MODEL_TYPE_TO_STATE_DICT_KEY_MAPPING,
)

DEFAULT_MAX_ENCODE_CHUNK_ELEMENTS = 2**30
"""Default budget, in tensor elements, for one chunk of :meth:`SpladeModel.encode`.

SPLADE is the only bi-encoder here whose per-token embedding is VOCABULARY-sized rather than
hidden-size-sized, so the intermediate it builds is ``[batch, seq_len, vocab_size]`` -- 40x a
768-dim model for BERT, 167x for deberta-v3. A reranking batch is one query's whole candidate
pool, and a Robust04 topic pool can exceed 2,400 documents, so that intermediate reaches
hundreds of GiB and no single card can hold it.

Chunking bounds it. The budget is expressed in ELEMENTS rather than documents so that one
value is correct for every backbone: the document count is derived per call as
``budget // (seq_len * vocab_size)``. A document count could not be, because the same memory
buys ~137 documents on BERT and ~32 on deberta-v3.

Elements convert to bytes at a factor of 8, which is the peak of the encode pipeline::

    e0 = projection(h)        [B,L,V] bf16  -.
    r  = relu(e0)             [B,L,V] bf16   +- live together, 8 B/element
    s  = log1p(r)             [B,L,V] fp32  -'                  (note: torch autocasts
    m  = s.masked_fill(...)   [B,L,V] fp32  -.                   log1p to fp32 on CUDA,
    out = m.amax(1)           [B,1,V]        +- live together, 8 B/element   which is why
                                            -'                   the fp32 rows appear)

So 2**30 elements is an ~8 GiB peak, which leaves ample headroom on a 40 GiB A100.
"""


def _slice_encoding(encoding: BatchEncoding, start: int, stop: int) -> BatchEncoding:
    """Slices the batch dimension of an already-padded encoding.

    Slicing the PADDED tensors rather than re-tokenizing is what keeps chunked encoding equivalent
    to unchunked encoding: every chunk keeps the batch's sequence length, so padding, position ids
    and the attention mask are bit-identical to what the full batch would have seen.

    Args:
        encoding (BatchEncoding): Tokenizer encodings for the text sequence.
        start (int): First sequence of the slice.
        stop (int): One past the last sequence of the slice.
    Returns:
        BatchEncoding: Encoding holding only the requested sequences.
    """
    batch_size = encoding["input_ids"].shape[0]
    return BatchEncoding(
        {
            key: value[start:stop] if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,) else value
            for key, value in encoding.items()
        }
    )


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
        max_encode_chunk_elements: int | None = DEFAULT_MAX_ENCODE_CHUNK_ELEMENTS,
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
            max_encode_chunk_elements (int | None): Budget, in tensor elements, for a single chunk of
                :meth:`SpladeModel.encode`. Only applies when gradients are disabled; training is never
                chunked. ``None`` disables chunking entirely. Defaults to
                :data:`DEFAULT_MAX_ENCODE_CHUNK_ELEMENTS`.
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
        self.max_encode_chunk_elements = max_encode_chunk_elements

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
        backbone_model_type = config.get_backbone_model_type() or config.model_type
        layer_cls = MODEL_TYPE_TO_LM_HEAD[backbone_model_type]
        self.projection = layer_cls(config)
        # Only register the decoder <-> input-embedding tie for backbones that actually tie the two.
        # NeoBERT pre-trains an *untied* MLM decoder (``tie_word_embeddings=False``); registering the
        # tie there would silently overwrite the pre-trained decoder with the word embeddings, which
        # is exactly the head SPLADE needs to start from.
        if getattr(config, "tie_word_embeddings", True):
            input_embeddings_key = MODEL_TYPE_TO_INPUT_EMBEDDINGS_KEY[backbone_model_type]
            tied_weights_keys = getattr(self, "_tied_weights_keys", None)
            if not isinstance(tied_weights_keys, dict):
                tied_weights_keys = {}
            else:
                tied_weights_keys = dict(tied_weights_keys)
            tied_weights_keys["projection.decoder.weight"] = input_embeddings_key
            self._tied_weights_keys = tied_weights_keys
            all_tied_weights_keys = getattr(self, "all_tied_weights_keys", None)
            if not isinstance(all_tied_weights_keys, dict):
                all_tied_weights_keys = {}
            else:
                all_tied_weights_keys = dict(all_tied_weights_keys)
            all_tied_weights_keys["projection.decoder.weight"] = input_embeddings_key
            self.all_tied_weights_keys = all_tied_weights_keys
        self.query_weights = None
        if config.query_weighting == "static":
            self.query_weights = torch.nn.Embedding(config.vocab_size, 1)
        self.doc_weights = None
        if config.doc_weighting == "static":
            self.doc_weights = torch.nn.Embedding(config.vocab_size, 1)

        self.pooler = Pooler(config)
        self.sparsifier = Sparsifier(config)

    def _encode_chunk_size(self, encoding: BatchEncoding) -> int | None:
        """Number of sequences to encode at a time, or None to encode the batch in one go.

        Derived from the configured element budget and the shape this call would actually build, so
        that a single configured value is correct for every backbone and every sequence length. See
        :data:`DEFAULT_MAX_ENCODE_CHUNK_ELEMENTS`.

        Args:
            encoding (BatchEncoding): Tokenizer encodings for the text sequence.
        Returns:
            int | None: Chunk size in sequences, or None if the batch should not be chunked.
        """
        budget = getattr(self.config, "max_encode_chunk_elements", None)
        if budget is None:
            return None
        # Chunking under autograd saves nothing -- every chunk's activations are kept alive for the
        # backward pass anyway -- and would additionally cost a cat. Skipping it here also means
        # training stays numerically identical to the unchunked implementation, so SPLADE runs
        # remain comparable to those produced before this change.
        if torch.is_grad_enabled():
            return None
        batch_size, seq_len = encoding["input_ids"].shape
        per_sequence = seq_len * self.config.embedding_dim
        chunk_size = max(1, budget // max(per_sequence, 1))
        if chunk_size >= batch_size:
            return None
        return chunk_size

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings and scoring mask.

        Encodes the batch in chunks when it would otherwise build a vocabulary-sized intermediate
        larger than the configured budget. Chunking is applied only when gradients are disabled, and
        is mathematically transparent: the already-padded encoding is sliced, so every chunk sees the
        same sequence length, attention never crosses sequences, and pooling is per-sequence.

        Args:
            encoding (BatchEncoding): Tokenizer encodings for the text sequence.
            input_type (Literal["query", "doc"]): type of input, either "query" or "doc".
        Returns:
            BiEncoderEmbedding: Embeddings and scoring mask.
        """
        chunk_size = self._encode_chunk_size(encoding)
        if chunk_size is None:
            return self._encode(encoding, input_type)
        batch_size = encoding["input_ids"].shape[0]
        embeddings = torch.cat(
            [
                self._encode(_slice_encoding(encoding, start, start + chunk_size), input_type).embeddings
                for start in range(0, batch_size, chunk_size)
            ],
            dim=0,
        )
        # The FULL encoding, not the last chunk's: callers treat this as the encoding that produced
        # the embeddings, and it is what an unchunked call would have returned.
        return BiEncoderEmbedding(embeddings, None, encoding)

    def _encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        """Encodes a single chunk of tokenized text sequences.

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
        if isinstance(config, type):
            backbone_model_type = getattr(config, "backbone_model_type", None)
            if backbone_model_type is None:
                for base in config.__mro__[1:]:
                    module_name = getattr(base, "__module__", "")
                    model_type = getattr(base, "model_type", None)
                    if module_name.startswith("transformers.models.") and model_type:
                        backbone_model_type = model_type
                        break
            model_type = backbone_model_type or getattr(config, "model_type", None)
        else:
            model_type = config.get_backbone_model_type() or config.model_type

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
