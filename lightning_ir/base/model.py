import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, Type, TypeVar

import torch
from transformers import MODEL_MAPPING, BatchEncoding, BertModel
from transformers.modeling_outputs import ModelOutput

from ..flash import FLASH_ATTENTION_MAP
from .class_factory import LightningIRModelClassFactory
from .config import LightningIRConfig
from .external_model_hub import CHECKPOINT_MAPPING, POST_LOAD_CALLBACKS, STATE_DICT_KEY_MAPPING


@dataclass
class LightningIROutput(ModelOutput):
    """Base class for the output of the LightningIR model. It is a subclass of transformers.ModelOutput_.

    .. _transformers.ModelOutput: https://huggingface.co/transformers/main_classes/output.html#transformers.ModelOutput

    :param scores: Output relevance scores for query--document pairs, defaults to None
    :type scores: torch.Tensor | None, optional
    """

    scores: torch.Tensor | None = None


class LightningIRModel:
    """Base class for LightningIR models. Derived classes implement the forward method for handling query
    and document embeddings. It acts as mixin for a transformers.PreTrainedModel_ backbone model.

    .. _transformers.PreTrainedModel: \
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel
    """

    config_class: Type[LightningIRConfig] = LightningIRConfig
    """Configuration class for the model."""

    ALLOW_SUB_BATCHING = True
    """Flag to allow mini batches of documents for a single query. Set to false for listwise models to ensure
    correctness."""

    def __init__(self, config: LightningIRConfig, *args, **kwargs) -> None:
        """Initializes the model.

        :param config: Configuration class for the model
        :type config: LightningIRConfig
        """
        super().__init__(config, *args, **kwargs)
        self.config = config

        self._sub_batch_size: int | None = None

        if self.config.backbone_model_type is not None:
            flash_attn = FLASH_ATTENTION_MAP.get(self.config.backbone_model_type, None)
            if flash_attn is not None:
                flash_attn_forward, self_attn_pattern = flash_attn
                for name, module in self.named_modules():
                    if name.endswith(self_attn_pattern):
                        module.forward = partial(flash_attn_forward, module)

    def _backbone_forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> LightningIROutput:
        """Forward method of the model. Must be implemented by the derived class."""
        raise NotImplementedError

    def _sparsification(
        self, embeddings: torch.Tensor, sparsification_strategy: Literal["relu", "relu_log"] | None = None
    ) -> torch.Tensor:
        """Helper method to apply sparsification to the embeddings.

        :param embeddings: Query or document embeddings
        :type embeddings: torch.Tensor
        :param sparsification_strategy: The sparsification strategy. No sparsification is applied if None,
        defaults to None
        :type sparsification_strategy: Literal[&quot;relu&quot;, &quot;relu_log&quot;] | None, optional
        :raises ValueError: If an unknown sparsification strategy is passed
        :return: (Optionally) sparsified embeddings
        :rtype: torch.Tensor
        """
        if sparsification_strategy is None:
            return embeddings
        if sparsification_strategy == "relu":
            return torch.relu(embeddings)
        if sparsification_strategy == "relu_log":
            return torch.log1p(torch.relu(embeddings))
        raise ValueError(f"Unknown sparsification strategy: {sparsification_strategy}")

    def _pooling(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None,
    ) -> torch.Tensor:
        """Helper method to apply pooling to the embeddings.

        :param embeddings: Query or document embeddings
        :type embeddings: torch.Tensor
        :param attention_mask: Query or document attention mask
        :type attention_mask: torch.Tensor | None
        :param pooling_strategy: The pooling strategy. No pooling is applied if None.
        :type pooling_strategy: Literal[&quot;first&quot;, &quot;mean&quot;, &quot;max&quot;, &quot;sum&quot;] | None
        :raises ValueError: If an unknown pooling strategy is passed
        :return: (Optionally) pooled embeddings
        :rtype: torch.Tensor
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
                embeddings = embeddings.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e9)
            return embeddings.max(dim=1, keepdim=True).values
        raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    @classmethod
    def _load_pretrained_model(
        cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
    ):
        if pretrained_model_name_or_path in STATE_DICT_KEY_MAPPING:
            map_keys = STATE_DICT_KEY_MAPPING[pretrained_model_name_or_path]
            for orig_key, new_key in map_keys:
                if orig_key is not None:
                    state_dict[new_key] = state_dict.pop(orig_key)
                    loaded_keys[loaded_keys.index(orig_key)] = new_key
                else:
                    loaded_keys.append(new_key)
        model, *out = super()._load_pretrained_model(
            model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
        )
        if pretrained_model_name_or_path in POST_LOAD_CALLBACKS:
            model = POST_LOAD_CALLBACKS[pretrained_model_name_or_path](model)
        return (model, *out)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path, *args, **kwargs) -> "LightningIRModel":
        """Loads a pretrained model. Wraps the transformers.PreTrainedModel.from_pretrained_ method and to return a
        derived LightningIRModel. See :class:`LightningIRModelClassFactory` for more details.

        .. _transformers.PreTrainedModel.from_pretrained: https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained # noqa

        :param model_name_or_path: Name or path of the pretrained model
        :type model_name_or_path: str | Path
        :raises ValueError: If called on the abstract class :class:`LightningIRModel` and no config is passed
        :return: A derived LightningIRModel consisting of a backbone model and a LightningIRModel mixin
        :rtype: LightningIRModel

        .. ::doctest
        .. highlight:: python
        .. code-block:: python

            >>> # Loading using model class and backbone checkpoint
            >>> type(CrossEncoderModel.from_pretrained("bert-base-uncased"))
            <class 'lightning_ir.base.class_factory.CrossEncoderBertModel'>
            >>> # Loading using base class and backbone checkpoint
            >>> type(LightningIRModel.from_pretrained("bert-base-uncased", config=CrossEncoderConfig()))
            <class 'lightning_ir.base.class_factory.CrossEncoderBertModel'>
        """
        # provides AutoModel.from_pretrained support
        config = kwargs.get("config", None)
        if cls is LightningIRModel or all(issubclass(base, LightningIRModel) for base in cls.__bases__):
            # no backbone models found, create derived lightning-ir model based on backbone model
            if model_name_or_path in CHECKPOINT_MAPPING:
                _config = CHECKPOINT_MAPPING[model_name_or_path]
                config_class = _config.__class__
                if config is not None:
                    warnings.warn(f"{model_name_or_path} is a registered checkpoint. The provided config is ignored.")
                config = _config
            elif config is not None:
                config_class = config.__class__
            elif cls is not LightningIRModel:
                config_class = cls.config_class
            else:
                config_class = LightningIRModelClassFactory.get_lightning_ir_config(model_name_or_path)
                if config_class is None:
                    raise ValueError("Pass a config to `from_pretrained`.")
            BackboneConfig = LightningIRModelClassFactory.get_backbone_config(model_name_or_path)
            BackboneModel = MODEL_MAPPING[BackboneConfig]
            cls = LightningIRModelClassFactory(config_class).from_backbone_class(BackboneModel)
            if config is not None and all(issubclass(base, LightningIRConfig) for base in config.__class__.__bases__):
                derived_config = cls.config_class.from_pretrained(model_name_or_path, config=config)
                derived_config.update(config.to_dict())
                kwargs["config"] = derived_config
            return cls.from_pretrained(model_name_or_path, *args, **kwargs)
        if issubclass(cls, BertModel):
            kwargs["add_pooling_layer"] = False
        return super(LightningIRModel, cls).from_pretrained(model_name_or_path, *args, **kwargs)


T = TypeVar("T")


def _cat_outputs(
    outputs: Sequence[Mapping] | Sequence[torch.Tensor] | Sequence[None], OutputClass: Type[T] | None
) -> torch.Tensor | T | None:
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
    return OutputClass(**{key: _cat_outputs(value, types[key]) for key, value in agg.items()})


def _batch_encoding(
    func: Callable[[LightningIRModel, BatchEncoding, ...], Any]
) -> Callable[[LightningIRModel, BatchEncoding, ...], Any]:

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
