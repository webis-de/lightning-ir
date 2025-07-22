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

        Args:
            MixinConfig (Type[LightningIRConfig]): LightningIRConfig mixin class.
        """
        if getattr(MixinConfig, "backbone_model_type", None) is not None:
            MixinConfig = MixinConfig.__bases__[0]
        self.MixinConfig = MixinConfig

    @staticmethod
    def get_backbone_config(model_name_or_path: str | Path) -> PretrainedConfig:
        """Grabs the configuration from a checkpoint of a pretrained HuggingFace model.

        Args:
            model_name_or_path (str | Path): Path to the model or its name.
        Returns:
            PretrainedConfig: Configuration of the backbone model.
        """
        backbone_model_type = LightningIRClassFactory.get_backbone_model_type(model_name_or_path)
        return CONFIG_MAPPING[backbone_model_type].from_pretrained(model_name_or_path)

    @staticmethod
    def get_lightning_ir_config(model_name_or_path: str | Path) -> LightningIRConfig | None:
        """Grabs the Lightning IR configuration from a checkpoint of a pretrained Lightning IR model.

        Args:
            model_name_or_path (str | Path): Path to the model or its name.
        Returns:
            LightningIRConfig | None: Configuration class of the Lightning IR model.
        """
        model_type = LightningIRClassFactory.get_lightning_ir_model_type(model_name_or_path)
        if model_type is None:
            return None
        return CONFIG_MAPPING[model_type].from_pretrained(model_name_or_path)

    @staticmethod
    def get_backbone_model_type(model_name_or_path: str | Path, *args, **kwargs) -> str:
        """Grabs the model type from a checkpoint of a pretrained HuggingFace model.

        Args:
            model_name_or_path (str | Path): Path to the model or its name.
        Returns:
            str: Model type of the backbone model.
        Raises:
            ValueError: If the type of the model is None in the configuration.
        """
        config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path, *args, **kwargs)
        backbone_model_type = config_dict.get("backbone_model_type", None) or config_dict.get("model_type")
        if backbone_model_type is None:
            raise ValueError(f"Unable to load PretrainedConfig from {model_name_or_path}")
        return backbone_model_type

    @staticmethod
    def get_lightning_ir_model_type(model_name_or_path: str | Path) -> str | None:
        """Grabs the Lightning IR model type from a checkpoint of a pretrained HuggingFace model.

        Args:
            model_name_or_path (str | Path): Path to the model or its name.
        Returns:
            str | None: Model type of the Lightning IR model.
        Raises:
            ValueError: If the backbone model type is not found in the configuration.
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

        Args:
            model_name_or_path (str | Path): Path to the model or its name.
        Returns:
            Any: Derived Lightning IR class.
        """
        ...

    @abstractmethod
    def from_backbone_class(self, BackboneClass: Type) -> Type:
        """Creates a derived Lightning IR class from a backbone HuggingFace class. Must be implemented by subclasses.

        Args:
            BackboneClass (Type): Backbone class.
        Returns:
            Type: Derived Lightning IR class.
        """
        ...


class LightningIRConfigClassFactory(LightningIRClassFactory):
    """Class factory for creating derived LightningIRConfig classes from HuggingFace configuration classes."""

    def from_pretrained(self, model_name_or_path: str | Path, *args, **kwargs) -> Type[LightningIRConfig]:
        """Loads a derived LightningIRConfig from a pretrained HuggingFace model.

        Args:
            model_name_or_path (str | Path): Path to the model or its name.
        Returns:
            Type[LightningIRConfig]: Derived LightningIRConfig.
        """
        backbone_config = self.get_backbone_config(model_name_or_path)
        DerivedLightningIRConfig = self.from_backbone_class(type(backbone_config))
        return DerivedLightningIRConfig

    def from_backbone_class(self, BackboneClass: Type[PretrainedConfig]) -> Type[LightningIRConfig]:
        """Creates a derived LightningIRConfig from a transformers.PretrainedConfig_ backbone configuration class. If
        the backbone configuration class is already a derived LightningIRConfig, it is returned as is.
        
        .. _transformers.PretrainedConfig: \
https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig

        Args:
            BackboneClass (Type[PretrainedConfig]): Backbone configuration class.
        Returns:
            Type[LightningIRConfig]: Derived LightningIRConfig.
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

        Args:
            model_name_or_path (str | Path): Path to the model or its name.
        Returns:
            Type[LightningIRModel]: Derived LightningIRModel.
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

        Args:
            BackboneClass (Type[PreTrainedModel]): Backbone model class.
        Returns:
            Type[LightningIRModel]: Derived LightningIRModel.
        Raises:
            ValueError: If the backbone model is not a valid backbone model.
            ValueError: If the backbone model is not a LightningIRModel and no LightningIRConfig is passed.
            ValueError: If the LightningIRModel mixin is not registered with the Hugging Face model mapping.
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

        Args:
            model_name_or_path (str | Path): Path to the tokenizer or its name.
        Returns:
            PretrainedConfig: Configuration class of the backbone tokenizer.
        """
        backbone_model_type = LightningIRTokenizerClassFactory.get_backbone_model_type(model_name_or_path)
        return CONFIG_MAPPING[backbone_model_type].from_pretrained(model_name_or_path)

    @staticmethod
    def get_backbone_model_type(model_name_or_path: str | Path, *args, **kwargs) -> str:
        """Grabs the model type from a checkpoint of a pretrained HuggingFace tokenizer.

        Args:
            model_name_or_path (str | Path): Path to the tokenizer or its name.
        Returns:
            str: Model type of the backbone tokenizer.
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

        Args:
            model_name_or_path (str | Path): Path to the tokenizer or its name.
            use_fast (bool, optional): Whether to use the fast tokenizer. Defaults to True.
        Returns:
            Type[LightningIRTokenizer]: Derived LightningIRTokenizer.
        Raises:
            ValueError: If no fast tokenizer is found when `use_fast` is True.
            ValueError: If no slow tokenizer is found when `use_fast` is False.
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

        Args:
            BackboneClasses (Tuple[Type[PreTrainedTokenizerBase] | None, Type[PreTrainedTokenizerBase] | None]):
                Slow and fast backbone tokenizer classes.
            BackboneConfig (Type[PretrainedConfig] | None, optional): Backbone configuration class. Defaults to None.
        Returns:
            Tuple[Type[LightningIRTokenizer] | None, Type[LightningIRTokenizer] | None]: Slow and fast derived
            LightningIRTokenizers.
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

        Args:
            BackboneClass (Type[PreTrainedTokenizerBase]): Backbone tokenizer class.
        Returns:
            Type[LightningIRTokenizer]: Derived LightningIRTokenizer.
        """
        if hasattr(BackboneClass, "config_class"):
            return BackboneClass
        LightningIRTokenizerMixin = TOKENIZER_MAPPING[self.MixinConfig][0]

        DerivedLightningIRTokenizer = type(
            f"{self.cc_lir_model_type}{BackboneClass.__name__}", (LightningIRTokenizerMixin, BackboneClass), {}
        )

        return DerivedLightningIRTokenizer
