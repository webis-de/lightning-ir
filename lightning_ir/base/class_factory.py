"""
Class factory module for Lightning IR.

This module provides factory classes for creating various components of the Lightning IR library
by extending Hugging Face Transformers classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple, Type

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    TOKENIZER_MAPPING,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config, tokenizer_class_from_name

if TYPE_CHECKING:
    from . import LightningIRConfig, LightningIRModel, LightningIRTokenizer


def _get_model_class(config: PretrainedConfig | Type[PretrainedConfig]) -> Type[PreTrainedModel]:
    # https://github.com/huggingface/transformers/blob/356b3cd71d7bfb51c88fea3e8a0c054f3a457ab9/src/transformers/models/auto/auto_factory.py#L387
    if isinstance(config, type):
        supported_models = MODEL_MAPPING[config]
    else:
        supported_models = MODEL_MAPPING[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    if isinstance(config, type):
        # we cannot parse architectures from a config class, we need an instance for this
        return supported_models[0]

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


class LightningIRClassFactory(ABC):
    """Base class for creating derived Lightning IR classes from HuggingFace classes."""

    def __init__(self, MixinConfig: Type[LightningIRConfig]) -> None:
        """Creates a new LightningIRClassFactory.

        :param MixinConfig: LightningIRConfig mixin class
        :type MixinConfig: Type[LightningIRConfig]
        """
        if getattr(MixinConfig, "backbone_model_type", None) is not None:
            MixinConfig = MixinConfig.__bases__[0]
        self.MixinConfig = MixinConfig

    @staticmethod
    def get_backbone_config(model_name_or_path: str | Path) -> PretrainedConfig:
        """Grabs the configuration from a checkpoint of a pretrained HuggingFace model.

        :param model_name_or_path: Path to the model or its name
        :type model_name_or_path: str | Path
        :return: Configuration of the backbone model
        :rtype: PretrainedConfig
        """
        backbone_model_type = LightningIRClassFactory.get_backbone_model_type(model_name_or_path)
        return CONFIG_MAPPING[backbone_model_type].from_pretrained(model_name_or_path)

    @staticmethod
    def get_lightning_ir_config(model_name_or_path: str | Path) -> LightningIRConfig | None:
        """Grabs the Lightning IR configuration from a checkpoint of a pretrained Lightning IR model.

        :param model_name_or_path: Path to the model or its name
        :type model_name_or_path: str | Path
        :return: Configuration class of the Lightning IR model
        :rtype: LightningIRConfig | None
        """
        model_type = LightningIRClassFactory.get_lightning_ir_model_type(model_name_or_path)
        if model_type is None:
            return None
        return CONFIG_MAPPING[model_type].from_pretrained(model_name_or_path)

    @staticmethod
    def get_backbone_model_type(model_name_or_path: str | Path, *args, **kwargs) -> str:
        """Grabs the model type from a checkpoint of a pretrained HuggingFace model.

        :param model_name_or_path: Path to the model or its name
        :type model_name_or_path: str | Path
        :return: Model type of the backbone model
        :rtype: str
        """
        config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path, *args, **kwargs)
        backbone_model_type = config_dict.get("backbone_model_type", None) or config_dict.get("model_type")
        if backbone_model_type is None:
            raise ValueError(f"Unable to load PretrainedConfig from {model_name_or_path}")
        return backbone_model_type

    @staticmethod
    def get_lightning_ir_model_type(model_name_or_path: str | Path) -> str | None:
        """Grabs the Lightning IR model type from a checkpoint of a pretrained HuggingFace model.

        :param model_name_or_path: Path to the model or its name
        :type model_name_or_path: str | Path
        :return: Model type of the Lightning IR model
        :rtype: str | None
        """
        config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
        if "backbone_model_type" not in config_dict:
            return None
        return config_dict.get("model_type", None)

    @property
    def cc_lir_model_type(self) -> str:
        """Camel case model type of the Lightning IR model."""
        return "".join(s.title() for s in self.MixinConfig.model_type.split("-"))

    @abstractmethod
    def from_pretrained(self, model_name_or_path: str | Path, *args, **kwargs) -> Any:
        """Loads a derived Lightning IR class from a pretrained HuggingFace model. Must be implemented by subclasses.

        :param model_name_or_path: Path to the model or its name
        :type model_name_or_path: str | Path
        :return: Derived Lightning IR class
        :rtype: Any
        """
        ...

    @abstractmethod
    def from_backbone_class(self, BackboneClass: Type) -> Type:
        """Creates a derived Lightning IR class from a backbone HuggingFace class. Must be implemented by subclasses.

        :param BackboneClass: Backbone class
        :type BackboneClass: Type
        :return: Derived Lightning IR class
        :rtype: Type
        """
        ...


class LightningIRConfigClassFactory(LightningIRClassFactory):
    """Class factory for creating derived LightningIRConfig classes from HuggingFace configuration classes."""

    def from_pretrained(self, model_name_or_path: str | Path, *args, **kwargs) -> Type[LightningIRConfig]:
        """Loads a derived LightningIRConfig from a pretrained HuggingFace model.

        :param model_name_or_path: Path to the model or its name
        :type model_name_or_path: str | Path
        :return: Derived LightningIRConfig
        :rtype: Type[LightningIRConfig]
        """
        backbone_config = self.get_backbone_config(model_name_or_path)
        DerivedLightningIRConfig = self.from_backbone_class(type(backbone_config))
        return DerivedLightningIRConfig

    def from_backbone_class(self, BackboneClass: Type[PretrainedConfig]) -> Type[LightningIRConfig]:
        """Creates a derived LightningIRConfig from a transformers.PretrainedConfig_ backbone configuration class. If
        the backbone configuration class is already a dervied LightningIRConfig, it is returned as is.

        .. _transformers.PretrainedConfig: \
https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig

        :param BackboneClass: Backbone configuration class
        :type BackboneClass: Type[PretrainedConfig]
        :return: Derived LightningIRConfig
        :rtype: Type[LightningIRConfig]
        """
        if getattr(BackboneClass, "backbone_model_type", None) is not None:
            return BackboneClass
        LightningIRConfigMixin: Type[LightningIRConfig] = CONFIG_MAPPING[self.MixinConfig.model_type]

        DerivedLightningIRConfig = type(
            f"{self.cc_lir_model_type}{BackboneClass.__name__}",
            (LightningIRConfigMixin, BackboneClass),
            {
                "model_type": self.MixinConfig.model_type,
                "backbone_model_type": BackboneClass.model_type,
                "mixin_config": self.MixinConfig,
            },
        )
        return DerivedLightningIRConfig


class LightningIRModelClassFactory(LightningIRClassFactory):
    """Class factory for creating derived LightningIRModel classes from HuggingFace model classes."""

    def from_pretrained(self, model_name_or_path: str | Path, *args, **kwargs) -> Type[LightningIRModel]:
        """Loads a derived LightningIRModel from a pretrained HuggingFace model.

        :param model_name_or_path: Path to the model or its name
        :type model_name_or_path: str | Path
        :return: Derived LightningIRModel
        :rtype: Type[LightningIRModel]
        """
        backbone_config = self.get_backbone_config(model_name_or_path)
        BackboneModel = _get_model_class(backbone_config)
        DerivedLightningIRModel = self.from_backbone_class(BackboneModel)
        return DerivedLightningIRModel

    def from_backbone_class(self, BackboneClass: Type[PreTrainedModel]) -> Type[LightningIRModel]:
        """Creates a derived LightningIRModel from a transformers.PreTrainedModel_ backbone model. If the backbone model
          is already a LightningIRModel, it is returned as is.

        .. _transformers.PreTrainedModel: \
https://huggingface.co/transformers/main_classes/model#transformers.PreTrainedModel

        :param BackboneClass: Backbone model
        :type BackboneClass: Type[PreTrainedModel]
        :raises ValueError: If the backbone model is not a valid backbone model.
        :raises ValueError: If the backbone model is not a LightningIRModel and no LightningIRConfig is passed.
        :raises ValueError: If the LightningIRModel mixin is not registered with the Hugging Face model mapping.
        :return: The derived LightningIRModel
        :rtype: Type[LightningIRModel]
        """
        if getattr(BackboneClass.config_class, "backbone_model_type", None) is not None:
            return BackboneClass
        BackboneConfig = BackboneClass.config_class
        if BackboneConfig is None:
            raise ValueError(
                f"Model {BackboneClass} is not a valid backbone model because it is missing a `config_class`."
            )

        LightningIRModelMixin: Type[LightningIRModel] = _get_model_class(self.MixinConfig)

        DerivedLightningIRConfig = LightningIRConfigClassFactory(self.MixinConfig).from_backbone_class(BackboneConfig)

        DerivedLightningIRModel = type(
            f"{self.cc_lir_model_type}{BackboneClass.__name__}",
            (LightningIRModelMixin, BackboneClass),
            {"config_class": DerivedLightningIRConfig, "_backbone_forward": BackboneClass.forward},
        )
        return DerivedLightningIRModel


class LightningIRTokenizerClassFactory(LightningIRClassFactory):
    """Class factory for creating derived LightningIRTokenizer classes from HuggingFace tokenizer classes."""

    @staticmethod
    def get_backbone_config(model_name_or_path: str | Path) -> PretrainedConfig:
        """Grabs the tokenizer configuration class from a checkpoint of a pretrained HuggingFace tokenizer.

        :param model_name_or_path: Path to the tokenizer or its name
        :type model_name_or_path: str | Path
        :return: Configuration class of the backbone tokenizer
        :rtype: PretrainedConfig
        """
        backbone_model_type = LightningIRTokenizerClassFactory.get_backbone_model_type(model_name_or_path)
        return CONFIG_MAPPING[backbone_model_type].from_pretrained(model_name_or_path)

    @staticmethod
    def get_backbone_model_type(model_name_or_path: str | Path, *args, **kwargs) -> str:
        """Grabs the model type from a checkpoint of a pretrained HuggingFace tokenizer.

        :param model_name_or_path: Path to the tokenizer or its name
        :type model_name_or_path: str | Path
        :return: Model type of the backbone tokenizer
        :rtype: str
        """
        try:
            return LightningIRClassFactory.get_backbone_model_type(model_name_or_path, *args, **kwargs)
        except (OSError, ValueError):
            # best guess at model type
            config_dict = get_tokenizer_config(model_name_or_path)
            backbone_tokenizer_class = config_dict.get("backbone_tokenizer_class", None)
            if backbone_tokenizer_class is not None:
                Tokenizer = tokenizer_class_from_name(backbone_tokenizer_class)
                for config, tokenizers in TOKENIZER_MAPPING.items():
                    if Tokenizer in tokenizers:
                        return getattr(config, "model_type")
            raise ValueError("No backbone model found in the configuration")

    def from_pretrained(
        self, model_name_or_path: str | Path, *args, use_fast: bool = True, **kwargs
    ) -> Type[LightningIRTokenizer]:
        """Loads a derived LightningIRTokenizer from a pretrained HuggingFace tokenizer.

        :param model_name_or_path: Path to the tokenizer or its name
        :type model_name_or_path: str | Path
        :param use_fast: Whether to use the fast or slow tokenizer, defaults to True
        :type use_fast: bool, optional
        :raises ValueError: If use_fast is True and no fast tokenizer is found
        :raises ValueError: If use_fast is False and no slow tokenizer is found
        :return: Derived LightningIRTokenizer
        :rtype: Type[LightningIRTokenizer]
        """
        backbone_config = self.get_backbone_config(model_name_or_path)
        BackboneTokenizers = TOKENIZER_MAPPING[type(backbone_config)]
        DerivedLightningIRTokenizers = self.from_backbone_classes(BackboneTokenizers, type(backbone_config))
        if use_fast:
            DerivedLightningIRTokenizer = DerivedLightningIRTokenizers[1]
            if DerivedLightningIRTokenizer is None:
                raise ValueError("No fast tokenizer found.")
        else:
            DerivedLightningIRTokenizer = DerivedLightningIRTokenizers[0]
            if DerivedLightningIRTokenizer is None:
                raise ValueError("No slow tokenizer found.")
        return DerivedLightningIRTokenizer

    def from_backbone_classes(
        self,
        BackboneClasses: Tuple[Type[PreTrainedTokenizerBase] | None, Type[PreTrainedTokenizerBase] | None],
        BackboneConfig: Type[PretrainedConfig] | None = None,
    ) -> Tuple[Type[LightningIRTokenizer] | None, Type[LightningIRTokenizer] | None]:
        """Creates derived slow and fastLightningIRTokenizers from a tuple of backbone HuggingFace tokenizer classes.

        :param BackboneClasses: Slow and fast backbone tokenizer classes
        :type BackboneClasses: Tuple[Type[PreTrainedTokenizerBase] | None, Type[PreTrainedTokenizerBase] | None]
        :param BackboneConfig: Backbone configuration class, defaults to None
        :type BackboneConfig: Type[PretrainedConfig], optional
        :return: Slow and fast derived LightningIRTokenizers
        :rtype: Tuple[Type[LightningIRTokenizer] | None, Type[LightningIRTokenizer] | None]
        """
        DerivedLightningIRTokenizers = tuple(
            None if BackboneClass is None else self.from_backbone_class(BackboneClass)
            for BackboneClass in BackboneClasses
        )
        if DerivedLightningIRTokenizers[1] is not None:
            DerivedLightningIRTokenizers[1].slow_tokenizer_class = DerivedLightningIRTokenizers[0]
        return DerivedLightningIRTokenizers

    def from_backbone_class(self, BackboneClass: Type[PreTrainedTokenizerBase]) -> Type[LightningIRTokenizer]:
        """Creates a derived LightningIRTokenizer from a transformers.PreTrainedTokenizerBase_ backbone tokenizer. If
        the backbone tokenizer is already a LightningIRTokenizer, it is returned as is.

        .. _transformers.PreTrainedTokenizerBase: \
https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizerBase

        :param BackboneClass: Backbone tokenizer class
        :type BackboneClass: Type[PreTrainedTokenizerBase]
        :return: Derived LightningIRTokenizer
        :rtype: Type[LightningIRTokenizer]
        """
        if hasattr(BackboneClass, "config_class"):
            return BackboneClass
        LightningIRTokenizerMixin = TOKENIZER_MAPPING[self.MixinConfig][0]

        DerivedLightningIRTokenizer = type(
            f"{self.cc_lir_model_type}{BackboneClass.__name__}", (LightningIRTokenizerMixin, BackboneClass), {}
        )

        return DerivedLightningIRTokenizer
