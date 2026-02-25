"""
Base configuration class for Lightning IR models.

This module defines the configuration class `LightningIRConfig` which is used to instantiate
a Lightning IR model. The configuration class acts as a mixin for the `transformers.PretrainedConfig`
class from the Hugging Face Transformers library.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig

from .class_factory import LightningIRConfigClassFactory
from .external_model_hub import CHECKPOINT_MAPPING

if TYPE_CHECKING:
    from .tokenizer import LightningIRTokenizer

    try:
        from peft import LoraConfig
    except ImportError:

        class LoraConfig:
            pass


class LightningIRConfig(PretrainedConfig):
    """The configuration class to instantiate a Lightning IR model. Acts as a mixin for the
    transformers.PretrainedConfig_ class.

    .. _transformers.PretrainedConfig: \
https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig
    """

    model_type = "lightning-ir"
    """Model type for the configuration."""
    backbone_model_type: str | None = None
    """Backbone model type for the configuration. set by :func:`LightningIRModelClassFactory`."""

    def __init__(
        self,
        *args,
        query_length: int | None = 32,
        doc_length: int | None = 512,
        use_adapter: bool = False,
        adapter_config: LoraConfig | None = None,
        pretrained_adapter_name_or_path: str | None = None,
        **kwargs,
    ):
        """Initializes the configuration.

        Args:
            query_length (int | None): Maximum number of tokens per query. If None does not truncate. Defaults to 32.
            doc_length (int | None): Maximum number of tokens per document. If None does not truncate. Defaults to 512.
            use_adapter (bool, optional): Whether to use LoRA adapters. Defaults to False.
            adapter_config (Optional[LoraConfig], optional): Configuration for LoRA adapters.
                Only used if use_adapter is True. Defaults to None.
            pretrained_adapter_name_or_path (Optional[str], optional): The path to a pretrained adapter to load.
                Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.query_length = query_length
        self.doc_length = doc_length
        self.use_adapter = use_adapter
        self.adapter_config = adapter_config
        self.pretrained_adapter_name_or_path = pretrained_adapter_name_or_path

    def get_tokenizer_kwargs(self, Tokenizer: type[LightningIRTokenizer]) -> dict[str, Any]:
        """Returns the keyword arguments for the tokenizer. This method is used to pass the configuration
        parameters to the tokenizer.

        Args:
            Tokenizer (type[LightningIRTokenizer]): Class of the tokenizer to be used.
        Returns:
            dict[str, Any]: Keyword arguments for the tokenizer.
        """
        return {k: getattr(self, k) for k in inspect.signature(Tokenizer.__init__).parameters if hasattr(self, k)}

    def to_dict(self) -> dict[str, Any]:
        """Overrides the transformers.PretrainedConfig.to_dict_ method to include the added arguments and the backbone
        model type.

        .. _transformers.PretrainedConfig.to_dict: \
https://huggingface.co/docs/transformers/en/main_classes/configuration#transformers.PretrainedConfig.to_dict

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        output = super().to_dict()
        if self.backbone_model_type is not None:
            output["backbone_model_type"] = self.backbone_model_type
        return output

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> LightningIRConfig:
        """Overrides transformers.PretrainedConfig.from_dict to handle transformers v5 behaviour where
        ``AutoConfig.from_pretrained`` directly calls ``from_dict`` on the registered config class,
        bypassing ``from_pretrained``. When called on a pure LightningIR mixin class (i.e. no backbone
        inheritance) and the ``config_dict`` contains a ``backbone_model_type`` key, the method builds
        the correct derived config (mixin + backbone) before delegating to the parent.

        Args:
            config_dict (dict): Dictionary used to instantiate the config.
        Returns:
            LightningIRConfig: Derived LightningIRConfig instance.
        """
        if (
            cls is LightningIRConfig or all(issubclass(base, LightningIRConfig) for base in cls.__bases__)
        ) and "backbone_model_type" in config_dict:
            from transformers import CONFIG_MAPPING

            backbone_model_type = config_dict["backbone_model_type"]
            try:
                BackboneConfig = CONFIG_MAPPING[backbone_model_type]
            except KeyError:
                BackboneConfig = None
            if BackboneConfig is not None and not issubclass(BackboneConfig, LightningIRConfig):
                ConfigClass = cls if cls is not LightningIRConfig else None
                if ConfigClass is None:
                    model_type = config_dict.get("model_type")
                    try:
                        ConfigClass = CONFIG_MAPPING[model_type] if model_type else cls
                    except KeyError:
                        ConfigClass = cls
                derived_cls = LightningIRConfigClassFactory(ConfigClass).from_backbone_class(BackboneConfig)
                return super(LightningIRConfig, derived_cls).from_dict(config_dict, **kwargs)
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *args, **kwargs) -> LightningIRConfig:
        """Loads the configuration from a pretrained model. Wraps the transformers.PretrainedConfig.from_pretrained_

        .. _transformers.PretrainedConfig.from_pretrained: \
https://huggingface.co/docs/transformers/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained

        Args:
            pretrained_model_name_or_path (str | Path): Pretrained model name or path.
        Returns:
            LightningIRConfig: Derived LightningIRConfig class.
        Raises:
            ValueError: If `pretrained_model_name_or_path` is not a Lightning IR model and no
                :py:class:`LightningIRConfig` is passed.
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
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
