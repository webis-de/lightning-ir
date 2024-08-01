from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Literal, Type

import torch
from transformers import CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, BertModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ..flash import FLASH_ATTENTION_MAP
from . import LightningIRConfig


@dataclass
class LightningIROutput(ModelOutput):
    """Base class for the output of the LightningIR model. It is a subclass of transformers.ModelOutput_.

    .. _transformers.ModelOutput: https://huggingface.co/transformers/main_classes/output.html#transformers.ModelOutput

    :param scores: Output relevance scores for query--document pairs, defaults to None
    :type scores: torch.Tensor | None, optional
    """

    scores: torch.Tensor | None = None


class LightningIRModel(ABC):
    """Base class for the LightningIR models. Derived classes implement the forward functionality for handling query
    and document embeddings. It acts as mixin for a transformers.PreTrainedModel_ backbone model.

    .. _transformers.PreTrainedModel: https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel
    """

    config_class: Type[LightningIRConfig] = LightningIRConfig
    """Configuration class for the model."""

    def __init__(self, config: LightningIRConfig, *args, **kwargs) -> None:
        """Initializes the model.

        :param config: Configuration class for the model
        :type config: LightningIRConfig
        """
        super().__init__(config, *args, **kwargs)
        self.config = config

        if self.config.backbone_model_type is not None:
            flash_attn = FLASH_ATTENTION_MAP.get(self.config.backbone_model_type, None)
            if flash_attn is not None:
                flash_attn_forward, self_attn_pattern = flash_attn
                for name, module in self.named_modules():
                    if name.endswith(self_attn_pattern):
                        module.forward = partial(flash_attn_forward, module)

    @abstractmethod
    def backbone_forward(self, *args, **kwargs):
        """Forward method of the backbone model. Is set by the :func:`LightningIRModelClassFactory`.

        :raises NotImplementedError: LightningIRModelClassFactory must set the backbone forward method
        """
        ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> LightningIROutput:
        """Forward method of the model. Must be implemented by the derived class."""
        ...

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
    def from_pretrained(cls, *args, **kwargs) -> "LightningIRModel":
        """Loads a pretrained model. Wraps the transformers.PreTrainedModel.from_pretrained_ method and returns a
        derived LightningIRModel. See :func:`LightningIRModelClassFactory` for more details.

        .. _transformers.PreTrainedModel.from_pretrained: https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

        .. highlight:: python
        .. code-block:: python

            >>> type(CrossEncoderModel.from_pretrained("bert-base-uncased"))
            ...
            <class 'lightning_ir.base.model.CrossEncoderBertModel'>
            >>> type(ColModel.from_pretrained("bert-base-uncased"))
            ...
            <class 'lightning_ir.base.model.ColBertModel'>

        :raises ValueError: If called on the abstract class :class:`LightningIRModel`.
        :raises ValueError: If the backbone model is not found.
        :return: A derived LightningIRModel consisting of a backbone model and a LightningIRModel mixin.
        :rtype: LightningIRModel
        """
        if cls is LightningIRModel:
            raise ValueError("LightningIRModel is an abstract class. Use either BiEncoderModel or CrossEncoderModel.")
        if all(issubclass(base, LightningIRModel) for base in cls.__bases__):
            # no backbone models found, create derived lightning-ir model based on backbone model
            config_dict, _ = PretrainedConfig.get_config_dict(*args, **kwargs)
            backbone_model_type = config_dict.get("backbone_model_type", None) or config_dict.get("model_type", None)
            if backbone_model_type is None:
                raise ValueError("No backbone model found in the configuration")
            try:
                BackboneModel = MODEL_MAPPING[CONFIG_MAPPING[backbone_model_type]]
            except KeyError:
                raise ValueError(f"Model {backbone_model_type} not found in the model mapping.")
            cls = LightningIRModelClassFactory(BackboneModel, cls.config_class)
        if issubclass(cls, BertModel):
            kwargs["add_pooling_layer"] = False
        return super(LightningIRModel, cls).from_pretrained(*args, **kwargs)


def LightningIRModelClassFactory(
    BackboneModel: Type[PreTrainedModel], MixinConfig: Type[LightningIRConfig] | None = None
) -> Type[LightningIRModel]:
    """Creates a derived LightningIRModel from a transformers.PreTrainedModel_ backbone model and a
    :class:LightningIRConfig mixin config. If the backbone model is already a LightningIRModel, it is returned as is.

    .. _transformers.PreTrainedModel: https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel

    :param BackboneModel: Backbone model
    :type BackboneModel: Type[PreTrainedModel]
    :param MixinConfig: LightningIR mixin config, defaults to None
    :type MixinConfig: Type[LightningIRConfig] | None, optional
    :raises ValueError: If the backbone model is not a valid backbone model.
    :raises ValueError: If the backbone model is not a LightningIRModel and no LightningIRConfig is passed.
    :raises ValueError: If the LightningIRModel mixin is not registered with the Hugging Face model mapping.
    :return: The derived LightningIRModel
    :rtype: Type[LightningIRModel]
    """
    if issubclass(BackboneModel, LightningIRModel):
        return BackboneModel

    BackboneConfig = BackboneModel.config_class
    if BackboneConfig is None or not issubclass(BackboneConfig, PretrainedConfig):
        raise ValueError(f"Model {BackboneModel} is not a valid backbone model because it is missing a `config_class`.")

    if MixinConfig is None or not issubclass(MixinConfig, LightningIRConfig):
        raise ValueError(f"Model {BackboneModel} is not a LightningIRModel, pass a LightningIRConfig to create one.")

    lir_model_type = MixinConfig.model_type
    LightningIRModelMixin: Type[LightningIRModel] | None = MODEL_MAPPING.get(MixinConfig, None)
    if LightningIRModelMixin is None:
        raise ValueError(f"Unable to find a LightningIRModel for config {MixinConfig.__name__}.")

    cc_lir_model_type = "".join(s.title() for s in lir_model_type.split("-"))
    cc_model_type = BackboneConfig.__name__[:-6]
    ModelConfig = type(
        f"{cc_lir_model_type}{cc_model_type}Config",
        (MixinConfig, BackboneConfig),
        {
            "backbone_model_type": BackboneConfig.model_type,
        },
    )

    DerivedLightningIRModel = type(
        f"{cc_lir_model_type}{cc_model_type}Model",
        (LightningIRModelMixin, BackboneModel),
        {"config_class": ModelConfig, "backbone_forward": BackboneModel.forward},
    )

    return DerivedLightningIRModel
