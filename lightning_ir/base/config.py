from pathlib import Path
from typing import Any, Dict, Set

from transformers import CONFIG_MAPPING

from .class_factory import LightningIRConfigClassFactory
from .external_model_hub import CHECKPOINT_MAPPING


class LightningIRConfig:
    """The configuration class to instantiate a LightningIR model. Acts as a mixin for the
    transformers.PretrainedConfig_ class.

    .. _transformers.PretrainedConfig: \
https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig
    """

    model_type = "lightning-ir"
    """Model type for the configuration."""
    backbone_model_type: str | None = None
    """Backbone model type for the configuration. Set by :func:`LightningIRModelClassFactory`."""

    TOKENIZER_ARGS: Set[str] = {"query_length", "doc_length"}
    """Arguments for the tokenizer."""
    ADDED_ARGS: Set[str] = TOKENIZER_ARGS
    """Arguments added to the configuration."""

    def __init__(self, *args, query_length: int = 32, doc_length: int = 512, **kwargs):
        """Initializes the configuration.

        :param query_length: Maximum query length, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum document length, defaults to 512
        :type doc_length: int, optional
        """
        super().__init__(*args, **kwargs)
        self.query_length = query_length
        self.doc_length = doc_length

    def to_added_args_dict(self) -> Dict[str, Any]:
        """Outputs a dictionary of the added arguments.

        :return: Added arguments
        :rtype: Dict[str, Any]
        """
        return {arg: getattr(self, arg) for arg in self.ADDED_ARGS if hasattr(self, arg)}

    def to_tokenizer_dict(self) -> Dict[str, Any]:
        """Outputs a dictionary of the tokenizer arguments.

        :return: Tokenizer arguments
        :rtype: Dict[str, Any]
        """
        return {arg: getattr(self, arg) for arg in self.TOKENIZER_ARGS}

    def to_dict(self) -> Dict[str, Any]:
        """Overrides the transformers.PretrainedConfig.to_dict_ method to include the added arguments and the backbone
        model type.

        .. _transformers.PretrainedConfig.to_dict: \
https://huggingface.co/docs/transformers/main_classes/configuration.html#transformers.PretrainedConfig.to_dict

        :return: Configuration dictionary
        :rtype: Dict[str, Any]
        """
        if hasattr(super(), "to_dict"):
            output = getattr(super(), "to_dict")()
        else:
            output = self.to_added_args_dict()
        if self.backbone_model_type is not None:
            output["backbone_model_type"] = self.backbone_model_type
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *args, **kwargs) -> "LightningIRConfig":
        if cls is LightningIRConfig or all(issubclass(base, LightningIRConfig) for base in cls.__bases__):
            config = None
            if pretrained_model_name_or_path in CHECKPOINT_MAPPING:
                config = CHECKPOINT_MAPPING[pretrained_model_name_or_path]
                config_class = config.__class__
            elif cls is not LightningIRConfig:
                config_class = cls
            else:
                config_class = LightningIRConfigClassFactory.get_lightning_ir_config(pretrained_model_name_or_path)
                if config_class is None:
                    raise ValueError("Pass a config to `from_pretrained`.")
            BackboneConfig = LightningIRConfigClassFactory.get_backbone_config(pretrained_model_name_or_path)
            cls = LightningIRConfigClassFactory(config_class).from_backbone_class(BackboneConfig)
            if config is not None and all(issubclass(base, LightningIRConfig) for base in config.__class__.__bases__):
                derived_config = cls.from_pretrained(pretrained_model_name_or_path, config=config)
                derived_config.update(config.to_dict())
            return cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return super(LightningIRConfig, cls).from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], *args, **kwargs) -> "LightningIRConfig":
        """Loads the configuration from a dictionary. Wraps the transformers.PretrainedConfig.from_dict_ method to
        return a derived LightningIRConfig class. See :class:`.LightningIRConfigClassFactory` for more details.

        .. _transformers.PretrainedConfig.from_dict: \
https://huggingface.co/docs/transformers/main_classes/configuration.html#transformers.PretrainedConfig.from_dict

        :param config_dict: Configuration dictionary
        :type config_dict: Dict[str, Any]
        :raises ValueError: If the model type does not match the configuration model type
        :return: Derived LightningIRConfig class
        :rtype: LightningIRConfig
        """
        if all(issubclass(base, LightningIRConfig) for base in cls.__bases__) or cls is LightningIRConfig:
            if "backbone_model_type" in config_dict:
                backbone_model_type = config_dict["backbone_model_type"]
                model_type = config_dict["model_type"]
                if cls is not LightningIRConfig and model_type != cls.model_type:
                    raise ValueError(
                        f"Model type {model_type} does not match configuration model type {cls.model_type}"
                    )
            else:
                backbone_model_type = config_dict["model_type"]
                model_type = cls.model_type
            MixinConfig = CONFIG_MAPPING[model_type]
            BackboneConfig = CONFIG_MAPPING[backbone_model_type]
            cls = LightningIRConfigClassFactory(MixinConfig).from_backbone_class(BackboneConfig)
            return cls.from_dict(config_dict, *args, **kwargs)
        return super(LightningIRConfig, cls).from_dict(config_dict, *args, **kwargs)
