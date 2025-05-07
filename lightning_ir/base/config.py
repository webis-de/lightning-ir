"""
Base configuration class for Lightning IR models.

This module defines the configuration class `LightningIRConfig` which is used to instantiate
a Lightning IR model. The configuration class acts as a mixin for the `transformers.PretrainedConfig`
class from the Hugging Face Transformers library.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Type

from transformers import PretrainedConfig

from .class_factory import LightningIRConfigClassFactory
from .external_model_hub import CHECKPOINT_MAPPING

if TYPE_CHECKING:
    from .tokenizer import LightningIRTokenizer


class LightningIRConfig(PretrainedConfig):
    """The configuration class to instantiate a Lightning IR model. Acts as a mixin for the
    transformers.PretrainedConfig_ class.

    .. _transformers.PretrainedConfig: \
https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig
    """

    model_type = "lightning-ir"
    """Model type for the configuration."""
    backbone_model_type: str | None = None
    """Backbone model type for the configuration. Set by :func:`LightningIRModelClassFactory`."""

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

    def get_tokenizer_kwargs(self, Tokenizer: Type[LightningIRTokenizer]) -> Dict[str, Any]:
        """Returns the keyword arguments for the tokenizer. This method is used to pass the configuration
        parameters to the tokenizer.

        :param Tokenizer: Class of the tokenizer to be used
        :type Tokenizer: Type[LightningIRTokenizer]
        :return: Keyword arguments for the tokenizer
        :rtype: Dict[str, Any]
        """
        return {k: getattr(self, k) for k in inspect.signature(Tokenizer.__init__).parameters if hasattr(self, k)}

    def to_dict(self) -> Dict[str, Any]:
        """Overrides the transformers.PretrainedConfig.to_dict_ method to include the added arguments and the backbone
        model type.

        .. _transformers.PretrainedConfig.to_dict: \
https://huggingface.co/docs/transformers/en/main_classes/configuration#transformers.PretrainedConfig.to_dict

        :return: Configuration dictionary
        :rtype: Dict[str, Any]
        """
        output = super().to_dict()
        if self.backbone_model_type is not None:
            output["backbone_model_type"] = self.backbone_model_type
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *args, **kwargs) -> "LightningIRConfig":
        """Loads the configuration from a pretrained model. Wraps the transformers.PretrainedConfig.from_pretrained_

        .. _transformers.PretrainedConfig.from_pretrained: \
https://huggingface.co/docs/transformers/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained

        :param pretrained_model_name_or_path: Pretrained model name or path
        :type pretrained_model_name_or_path: str | Path
        :raises ValueError: If `pre_trained_model_name_or_path` is not a Lightning IR model and no
            :py:class:`LightningIRConfig` is passed
        :return: Derived LightningIRConfig class
        :rtype: LightningIRConfig
        """
        # provides AutoConfig.from_pretrained support
        if cls is LightningIRConfig or all(issubclass(base, LightningIRConfig) for base in cls.__bases__):
            # no backbone config found, create derived lightning-ir config based on backbone config
            config = None
            if pretrained_model_name_or_path in CHECKPOINT_MAPPING:
                config = CHECKPOINT_MAPPING[pretrained_model_name_or_path]
                ConfigClass = config.__class__
            elif cls is not LightningIRConfig:
                ConfigClass = cls
            else:
                ConfigClass = type(LightningIRConfigClassFactory.get_lightning_ir_config(pretrained_model_name_or_path))
                if ConfigClass is None:
                    raise ValueError("Pass a config to `from_pretrained`.")
            backbone_config = LightningIRConfigClassFactory.get_backbone_config(pretrained_model_name_or_path)
            cls = LightningIRConfigClassFactory(ConfigClass).from_backbone_class(type(backbone_config))
            if config is not None and all(issubclass(base, LightningIRConfig) for base in config.__class__.__bases__):
                derived_config = cls.from_pretrained(pretrained_model_name_or_path, config=config)
                derived_config.update(config.to_dict())
            return cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return super(LightningIRConfig, cls).from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
