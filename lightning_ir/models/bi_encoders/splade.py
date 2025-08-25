"""Configuration and model for SPLADE (SParse Lexical AnD Expansion) type models. Originally proposed in
`SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking
<https://dl.acm.org/doi/abs/10.1145/3404835.3463098>`_.
"""

import warnings
from pathlib import Path
from typing import Literal, Self

import torch
from transformers import BatchEncoding

from ...bi_encoder import BiEncoderEmbedding, SingleVectorBiEncoderConfig, SingleVectorBiEncoderModel
from ...modeling_utils.mlm_head import (
    MODEL_TYPE_TO_KEY_MAPPING,
    MODEL_TYPE_TO_LM_HEAD,
    MODEL_TYPE_TO_OUTPUT_EMBEDDINGS,
    MODEL_TYPE_TO_TIED_WEIGHTS_KEYS,
)


class SpladeConfig(SingleVectorBiEncoderConfig):
    """Configuration class for a SPLADE model."""

    model_type = "splade"
    """Model type for a SPLADE model."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        sparsification: Literal["relu", "relu_log"] | None = "relu_log",
        query_pooling_strategy: Literal["first", "mean", "max", "sum"] = "max",
        doc_pooling_strategy: Literal["first", "mean", "max", "sum"] = "max",
        **kwargs,
    ) -> None:
        """A SPLADE model encodes queries and documents separately. Before computing the similarity score, the
        contextualized token embeddings are projected into a logit distribution over the vocabulary using a pre-trained
        masked language model (MLM) head. The logit distribution is then sparsified and aggregated to obtain a single
        embedding for the query and document.

        Args:
            query_length (int): Maximum query length. Defaults to 32.
            doc_length (int): Maximum document length. Defaults to 512.
            similarity_function (Literal["cosine", "dot"]): Similarity function to compute scores between query and
                document embeddings. Defaults to "dot".
            sparsification (Literal["relu", "relu_log"] | None): Sparsification function to apply.
                Defaults to "relu_log".
            query_pooling_strategy (Literal["first", "mean", "max", "sum"]): Pooling strategy for query embeddings.
                Defaults to "max".
            doc_pooling_strategy (Literal["first", "mean", "max", "sum"]): Pooling strategy for document embeddings.
                Defaults to "max".
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            sparsification=sparsification,
            query_pooling_strategy=query_pooling_strategy,
            doc_pooling_strategy=doc_pooling_strategy,
            **kwargs,
        )

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
        tied_weight_keys = getattr(self, "_tied_weights_keys", []) or []
        tied_weight_keys = tied_weight_keys + [
            f"projection.{key}"
            for key in MODEL_TYPE_TO_TIED_WEIGHTS_KEYS[config.backbone_model_type or config.model_type]
        ]
        setattr(self, "_tied_weights_keys", tied_weight_keys)

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings and scoring mask.

        Args:
            encoding (BatchEncoding): Tokenizer encodings for the text sequence.
            input_type (Literal["query", "doc"]): Type of input, either "query" or "doc".
        Returns:
            BiEncoderEmbedding: Embeddings and scoring mask.
        """
        pooling_strategy = getattr(self.config, f"{input_type}_pooling_strategy")
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.projection(embeddings)
        embeddings = self.sparsification(embeddings, self.config.sparsification)
        embeddings = self.pooling(embeddings, encoding["attention_mask"], pooling_strategy)
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
        if model_type in MODEL_TYPE_TO_KEY_MAPPING:
            key_mapping.update(MODEL_TYPE_TO_KEY_MAPPING[model_type])
        if not key_mapping:
            warnings.warn(
                f"No mlm key mappings for model_type {model_type} were provided. "
                "The pre-trained mlm weights will not be loaded correctly."
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
        module_names = MODEL_TYPE_TO_OUTPUT_EMBEDDINGS[self.config.backbone_model_type or self.config.model_type]
        output = self.projection
        for module_name in module_names.split("."):
            output = getattr(output, module_name)
        return output
