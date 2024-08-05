from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Tuple, Type

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    TOKENIZER_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config, tokenizer_class_from_name

if TYPE_CHECKING:
    from . import LightningIRConfig, LightningIRModel, LightningIRTokenizer


class LightningIRClassFactory(ABC):

    def __init__(self, MixinConfig: Type[LightningIRConfig]) -> None:
        if getattr(MixinConfig, "backbone_model_type", None) is not None:
            MixinConfig = MixinConfig.__bases__[0]
        self.MixinConfig = MixinConfig

    @staticmethod
    def get_backbone_config(model_name_or_path: str) -> PretrainedConfig:
        backbone_model_type = LightningIRClassFactory.get_backbone_model_type(model_name_or_path)
        return CONFIG_MAPPING[backbone_model_type]

    @staticmethod
    def get_backbone_model_type(model_name_or_path: str, *args, **kwargs) -> str:
        config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path, *args, **kwargs)
        backbone_model_type = config_dict.get("backbone_model_type", None) or config_dict.get("model_type", None)
        if backbone_model_type is None:
            raise ValueError("No backbone model found in the configuration")
        return backbone_model_type

    @property
    def cc_lir_model_type(self) -> str:
        return "".join(s.title() for s in self.MixinConfig.model_type.split("-"))

    @abstractmethod
    def from_pretrained(self, model_name_or_path: str, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def from_backbone_class(self, BackboneClass: Type) -> Type:
        pass


class LightningIRConfigClassFactory(LightningIRClassFactory):

    def from_pretrained(self, model_name_or_path: str, *args, **kwargs) -> Type[LightningIRConfig]:
        BackboneConfig = self.get_backbone_config(model_name_or_path)
        DerivedLightningIRConfig = self.from_backbone_class(BackboneConfig)
        return DerivedLightningIRConfig

    def from_backbone_class(self, BackboneClass: Type[PretrainedConfig]) -> Type[LightningIRConfig]:
        if getattr(BackboneClass, "backbone_model_type", None) is not None:
            return BackboneClass
        LightningIRConfigMixin: Type[LightningIRConfig] = CONFIG_MAPPING[self.MixinConfig.model_type]

        DerivedLightningIRConfig = type(
            f"{self.cc_lir_model_type}{BackboneClass.__name__}",
            (LightningIRConfigMixin, BackboneClass),
            {
                "model_type": f"{BackboneClass.model_type}-{self.MixinConfig.model_type}",
                "backbone_model_type": BackboneClass.model_type,
            },
        )

        AutoConfig.register(DerivedLightningIRConfig.model_type, DerivedLightningIRConfig, exist_ok=True)

        return DerivedLightningIRConfig


class LightningIRModelClassFactory(LightningIRClassFactory):

    def from_pretrained(self, model_name_or_path: str, *args, **kwargs) -> Type[LightningIRModel]:
        BackboneConfig = self.get_backbone_config(model_name_or_path)
        BackboneModel = MODEL_MAPPING[BackboneConfig]
        DerivedLightningIRModel = self.from_backbone_class(BackboneModel)
        return DerivedLightningIRModel

    def from_backbone_class(self, BackboneClass: Type[PreTrainedModel]) -> Type[LightningIRModel]:
        """Creates a derived LightningIRModel from a transformers.PreTrainedModel_ backbone model. If the backbone model
          is already a LightningIRModel, it is returned as is.

        .. _transformers.PreTrainedModel: https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel

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

        LightningIRModelMixin: Type[LightningIRModel] = MODEL_MAPPING[self.MixinConfig]

        DerivedLightningIRConfig = LightningIRConfigClassFactory(self.MixinConfig).from_backbone_class(BackboneConfig)

        DerivedLightningIRModel = type(
            f"{self.cc_lir_model_type}{BackboneClass.__name__}",
            (LightningIRModelMixin, BackboneClass),
            {"config_class": DerivedLightningIRConfig, "backbone_forward": BackboneClass.forward},
        )

        AutoModel.register(DerivedLightningIRConfig, DerivedLightningIRModel, exist_ok=True)

        return DerivedLightningIRModel


class LightningIRTokenizerClassFactory(LightningIRClassFactory):

    @staticmethod
    def get_backbone_config(model_name_or_path: str) -> PretrainedConfig:
        backbone_model_type = LightningIRTokenizerClassFactory.get_backbone_model_type(model_name_or_path)
        return CONFIG_MAPPING[backbone_model_type]

    @staticmethod
    def get_backbone_model_type(model_name_or_path: str, *args, **kwargs) -> str:
        try:
            return LightningIRClassFactory.get_backbone_model_type(model_name_or_path, *args, **kwargs)
        except OSError:
            # best guess at model type
            config_dict = get_tokenizer_config(model_name_or_path)
            Tokenizer = tokenizer_class_from_name(config_dict["tokenizer_class"])
            for config, tokenizers in TOKENIZER_MAPPING.items():
                if Tokenizer in tokenizers:
                    return getattr(config, "backbone_model_type", None) or getattr(config, "model_type")
            raise ValueError("No backbone model found in the configuration")

        # config_dict = get_tokenizer_config(model_name_or_path)

    def from_pretrained(
        self, model_name_or_path: str, *args, use_fast: bool = True, **kwargs
    ) -> Type[LightningIRTokenizer]:
        BackboneConfig = self.get_backbone_config(model_name_or_path)
        BackboneTokenizers = TOKENIZER_MAPPING[BackboneConfig]
        DerivedLightningIRTokenizers = self.from_backbone_classes(BackboneTokenizers, BackboneConfig)
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
        DerivedLightningIRTokenizers = tuple(
            None if BackboneClass is None else self.from_backbone_class(BackboneClass)
            for BackboneClass in BackboneClasses
        )
        if DerivedLightningIRTokenizers[1] is not None:
            DerivedLightningIRTokenizers[1].slow_tokenizer_class = DerivedLightningIRTokenizers[0]
        DerivedLightningIRConfig = LightningIRConfigClassFactory(self.MixinConfig).from_backbone_class(BackboneConfig)
        AutoTokenizer.register(
            DerivedLightningIRConfig, DerivedLightningIRTokenizers[0], DerivedLightningIRTokenizers[1]
        )
        return DerivedLightningIRTokenizers

    def from_backbone_class(self, BackboneClass: Type[PreTrainedTokenizerBase]) -> Type[LightningIRTokenizer]:
        if hasattr(BackboneClass, "config_class"):
            return BackboneClass
        LightningIRTokenizerMixin = TOKENIZER_MAPPING[self.MixinConfig][0]

        DerivedLightningIRTokenizer = type(
            f"{self.cc_lir_model_type}{BackboneClass.__name__}", (LightningIRTokenizerMixin, BackboneClass), {}
        )

        return DerivedLightningIRTokenizer
