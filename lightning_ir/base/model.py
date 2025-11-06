"""
Model module for Lightning IR.

This module contains the main model class and output class for the Lightning IR library.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Self, Sequence, Type, TypeVar

import torch
from transformers import BatchEncoding, BertModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .adapter import LightningIRAdapterMixin
from .class_factory import LightningIRModelClassFactory, _get_model_class
from .config import LightningIRConfig
from .external_model_hub import BACKBONE_MAPPING, CHECKPOINT_MAPPING, POST_LOAD_CALLBACKS, STATE_DICT_KEY_MAPPING


def _update_config_with_kwargs(config: LightningIRConfig, **kwargs):
    config.update(kwargs)

    used_keys = set(config.to_dict().keys()) & set(kwargs.keys())

    for key in used_keys:
        kwargs.pop(key)

    return config, kwargs


@dataclass
class LightningIROutput(ModelOutput):
    """Base class for the output of the Lightning IR model. It is a subclass of transformers.ModelOutput_.

    .. _transformers.ModelOutput: https://huggingface.co/transformers/main_classes/output.html#transformers.ModelOutput

    Attributes:
        scores (torch.Tensor | None): Output relevance scores for query--document pairs. Defaults to None.
    """

    scores: torch.Tensor | None = None


class LightningIRModel(LightningIRAdapterMixin, PreTrainedModel):
    """Base class for Lightning IR models. Derived classes implement the forward method for handling query
    and document embeddings. It acts as mixin for a transformers.PreTrainedModel_ backbone model.

    .. _transformers.PreTrainedModel: \
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel

    Attributes:
        config_class (Type[LightningIRConfig]): Configuration class for the model.
        ALLOW_SUB_BATCHING (bool): Flag to allow mini batches of documents for a single query.
            Set to false for listwise models to ensure correctness.
    """

    config_class: Type[LightningIRConfig] = LightningIRConfig
    """Configuration class for the model."""

    ALLOW_SUB_BATCHING = True
    """Flag to allow mini batches of documents for a single query. Set to false for listwise models to ensure
    correctness."""

    def __init__(self, config: LightningIRConfig, *args, **kwargs) -> None:
        """Initializes the model.

        Args:
            config(LightningIRConfig): Configuration class for the model
        """
        super().__init__(config, *args, **kwargs)
        self.config = config

        self._sub_batch_size: int | None = None

    def _initialize_adapters(self) -> None:
        """Initialize adapters based on configuration."""
        if not self.config.use_adapter:
            return

        # Enable adapters if configuration is provided
        if self.config.adapter_config is not None:
            self.init_adapters(self.config.adapter_config)

        # Load adapter weights if path is provided
        if self.config.pretrained_adapter_name_or_path is not None:
            self.load_adapter(self.config.pretrained_adapter_name_or_path)

    def _backbone_forward(self, *args, **kwargs):
        """Runs the forward method of the backbone model. Is overridden in
        :class:`~lightning_ir.base.class_factory.LightningIRModelClassFactory`.

        Raises:
            NotImplementedError: If not overridden in the derived class
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> LightningIROutput:
        """Forward method of the model. Must be implemented by the derived class."""
        raise NotImplementedError

    def sparsification(
        self, embeddings: torch.Tensor, sparsification_strategy: Literal["relu", "relu_log", "relu_2xlog"] | None = None
    ) -> torch.Tensor:
        """Helper method to apply sparsification to the embeddings.

        Args:
            embeddings(torch.Tensor): Query or document embeddings
            sparsification_strategy(Literal['relu', 'relu_log', 'relu_2xlog'] | None): The sparsification strategy. No
                sparsification is applied if None. Defaults to None.
        Returns:
            torch.Tensor: (Optionally) sparsified embeddings.
        Raises:
            ValueError: If an unknown sparsification strategy is passed.
        """
        if sparsification_strategy is None:
            return embeddings
        if sparsification_strategy == "relu":
            return torch.relu(embeddings)
        if sparsification_strategy == "relu_log":
            return torch.log1p(torch.relu(embeddings))
        if sparsification_strategy == "relu_2xlog":
            return torch.log1p(torch.log1p(torch.relu(embeddings)))
        raise ValueError(f"Unknown sparsification strategy: {sparsification_strategy}")

    def pooling(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None,
    ) -> torch.Tensor:
        """Helper method to apply pooling to the embeddings.

        Args:
            embeddings (torch.Tensor): Query or document embeddings
            attention_mask (torch.Tensor | None): Query or document attention mask
            pooling_strategy (Literal['first', 'mean', 'max', 'sum'] | None):
                The pooling strategy. No pooling is applied if None.
        Returns:
            torch.Tensor: (Optionally) pooled embeddings.
        Raises:
            ValueError: If an unknown pooling strategy is passed.
        """
        if pooling_strategy is None:
            return embeddings
        if pooling_strategy == "first":
            return embeddings[:, [0]]
        if pooling_strategy in ("sum", "mean"):
            if attention_mask is not None:
                embeddings = embeddings * attention_mask.unsqueeze(-1)
            embeddings = embeddings.sum(dim=1, keepdim=True)
            if pooling_strategy == "mean":
                if attention_mask is not None:
                    embeddings = embeddings / attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
            return embeddings
        if pooling_strategy == "max":
            if attention_mask is not None:
                embeddings = embeddings.masked_fill(~attention_mask.bool().unsqueeze(-1), float("-inf"))
            return embeddings.amax(dim=1, keepdim=True)
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str | Path, *args, BackboneModel: Type[PreTrainedModel] | None = None, **kwargs
    ) -> Self:
        """Loads a pretrained model. Wraps the transformers.PreTrainedModel.from_pretrained_ method to return a
        derived LightningIRModel. See :class:`LightningIRModelClassFactory` for more details.

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
            BackboneModel (Type[PreTrainedModel] | None): Huggingface PreTrainedModel class to use as backbone
                instead of the default AutoModel. Defaults to None.
        Raises:
            ValueError: If called on the abstract class `LightningIRModel` and no config is passed.
        Returns:
            LightningIRModel: A derived `LightningIRModel` consisting of a backbone model
            and a `LightningIRModel` mixin.
        """
        # provides AutoModel.from_pretrained support
        config = kwargs.get("config", None)
        if cls is LightningIRModel or all(issubclass(base, LightningIRModel) for base in cls.__bases__):
            # no backbone models found, create derived lightning-ir model based on backbone model
            if config is not None:
                ConfigClass = config.__class__
            elif model_name_or_path in CHECKPOINT_MAPPING:
                _config = CHECKPOINT_MAPPING[model_name_or_path]
                ConfigClass = _config.__class__
                if config is None:
                    config = _config
            elif cls is not LightningIRModel:
                ConfigClass = cls.config_class
            else:
                ConfigClass = type(LightningIRModelClassFactory.get_lightning_ir_config(model_name_or_path))
                if ConfigClass is None:
                    raise ValueError("Pass a config to `from_pretrained`.")
            if BackboneModel is None:
                if model_name_or_path in BACKBONE_MAPPING:
                    BackboneModel = BACKBONE_MAPPING[str(model_name_or_path)]
                else:
                    backbone_config = LightningIRModelClassFactory.get_backbone_config(
                        model_name_or_path
                    ).from_pretrained(model_name_or_path)
                    BackboneModel = _get_model_class(backbone_config)
            cls = LightningIRModelClassFactory(ConfigClass).from_backbone_class(BackboneModel)
            if config is not None:
                if all(issubclass(base, LightningIRConfig) for base in config.__class__.__bases__):
                    derived_config = cls.config_class.from_pretrained(model_name_or_path, config=config)
                    derived_config.update(config.to_diff_dict())
                    config = derived_config
                    kwargs["config"] = config
                # NOTE 'config' is contained in kwargs, so we can update it
                config, kwargs = _update_config_with_kwargs(**kwargs)
                kwargs["config"] = config
            return cls.from_pretrained(model_name_or_path, *args, **kwargs)
        if issubclass(cls, BertModel):
            kwargs["add_pooling_layer"] = False
        key_mapping = kwargs.pop("key_mapping", {})
        if model_name_or_path in STATE_DICT_KEY_MAPPING:
            key_mapping.update(STATE_DICT_KEY_MAPPING[str(model_name_or_path)])
        model = super().from_pretrained(model_name_or_path, *args, key_mapping=key_mapping, **kwargs)
        if model_name_or_path in POST_LOAD_CALLBACKS:
            model = POST_LOAD_CALLBACKS[str(model_name_or_path)](model)

        # Initialize adapters after model is fully loaded
        model._initialize_adapters()

        return model


T = TypeVar("T")


def _cat_outputs(
    outputs: Sequence[Mapping] | Sequence[torch.Tensor] | Sequence[None], OutputClass: Type[T] | None
) -> torch.Tensor | T | None:
    """Helper method to concatenate outputs of the model.

    Args:
        outputs (Sequence[Mapping] | Sequence[torch.Tensor] | Sequence[None]): Outputs from the model.
        OutputClass (Type[T] | None): Class to return the concatenated output as.
    Returns:
        torch.Tensor | T | None: Concatenated output.
    """
    if len(outputs) == 1:
        return outputs[0]
    if len(outputs) == 0 or outputs[0] is None or OutputClass is None:
        return None
    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs, dim=0)
    agg = defaultdict(list)
    types = {}
    for output in outputs:
        for key, value in output.items():
            agg[key].append(value)
            types[key] = type(value)
    kwargs = {key: _cat_outputs(value, types[key]) for key, value in agg.items()}
    if OutputClass is BatchEncoding:
        return OutputClass(kwargs)
    return OutputClass(**kwargs)


class BatchEncodingWrapper(Protocol):
    def __call__(self, encoding: BatchEncoding, *args, **kwargs) -> Any: ...


def batch_encoding_wrapper(func: BatchEncodingWrapper) -> BatchEncodingWrapper:
    """Decorator to enable sub-batching for models that support it. Lowers the batch size of the input batch encoding
    if the model runs out of memory.

    Args:
        func (BatchEncodingWrapper): Function to wrap that takes a batch encoding.
    Returns:
        BatchEncodingWrapper: Wrapped function that handles sub-batching.
    Raises:
        RuntimeError: If CUDA runs out of memory and the batch size cannot be lowered further.
        ValueError: If no output was generated.
    """

    @wraps(func)
    def wrapper(self, encoding: BatchEncoding, *args, **kwargs) -> Any:
        if not self.ALLOW_SUB_BATCHING:
            return func(self, encoding, *args, **kwargs)
        sub_batch_size = self._sub_batch_size or encoding.input_ids.shape[0]
        sub_encoding = encoding
        remaining_encoding = encoding
        OutputClass = None
        outputs = []
        while True:
            try:
                # ceil division
                num_batches = -(remaining_encoding.input_ids.shape[0] // -sub_batch_size)
                for _ in range(num_batches):
                    sub_encoding = BatchEncoding(
                        {key: value[:sub_batch_size] for key, value in remaining_encoding.items()}
                    )
                    output = func(self, sub_encoding, *args, **kwargs)
                    OutputClass = output.__class__
                    outputs.append(output)
                    remaining_encoding = BatchEncoding(
                        {key: value[sub_batch_size:] for key, value in remaining_encoding.items()}
                    )
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "CUDACachingAllocator.cpp" in str(e):
                    self._sub_batch_size = sub_batch_size = sub_batch_size // 2
                    if sub_batch_size == 0:
                        raise e
                else:
                    raise e
        if OutputClass is None:
            raise ValueError("No output was generated.")
        return _cat_outputs(outputs, OutputClass)

    return wrapper
