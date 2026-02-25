"""
Tokenizer module for Lightning IR.

This module contains the main tokenizer class for the Lightning IR library.
"""

import json
from collections.abc import Sequence
from os import PathLike
from typing import Self

from transformers import BatchEncoding, PreTrainedTokenizerBase

from .class_factory import LightningIRTokenizerClassFactory
from .config import LightningIRConfig
from .external_model_hub import CHECKPOINT_MAPPING


class LightningIRTokenizer(PreTrainedTokenizerBase):
    """Base class for Lightning IR tokenizers. Derived classes implement the tokenize method for handling query
    and document tokenization. It acts as mixin for a transformers.PreTrainedTokenizer_ backbone tokenizer.

    .. _transformers.PreTrainedTokenizer: \
https://huggingface.co/transformers/main_classes/tokenizer.htmltransformers.PreTrainedTokenizer
    """

    config_class: type[LightningIRConfig] = LightningIRConfig
    """Configuration class for the tokenizer."""

    def __init__(self, *args, query_length: int | None = 32, doc_length: int | None = 512, **kwargs):
        """Initializes the tokenizer.

        Args:
            query_length (int | None): Maximum number of tokens per query. If None does not truncate. Defaults to 32.
            doc_length (int | None): Maximum number of tokens per document. If None does not truncate. Defaults to 512.
        """
        super().__init__(*args, query_length=query_length, doc_length=doc_length, **kwargs)
        self.query_length = query_length
        self.doc_length = doc_length

    def tokenize(
        self, queries: str | Sequence[str] | None = None, docs: str | Sequence[str] | None = None, **kwargs
    ) -> dict[str, BatchEncoding]:
        """Tokenizes queries and documents.

        Args:
            queries (str | Sequence[str] | None): Queries to tokenize. Defaults to None.
            docs (str | Sequence[str] | None): Documents to tokenize. Defaults to None.
        Returns:
            dict[str, BatchEncoding]: Dictionary containing tokenized queries and documents.
        Raises:
            NotImplementedError: Must be implemented by the derived class.
        """
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args, **kwargs) -> Self:
        """Loads a pretrained tokenizer. Wraps the transformers.PreTrainedTokenizer.from_pretrained_ method to return a
        derived LightningIRTokenizer class. See :class:`.LightningIRTokenizerClassFactory` for more details.

        .. _transformers.PreTrainedTokenizer.from_pretrained: \
https://huggingface.co/docs/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.from_pretrained

        .. highlight:: python
        .. code-block:: python

            >>> Loading using model class and backbone checkpoint
            >>> type(BiEncoderTokenizer.from_pretrained("bert-base-uncased"))
            ...
            <class 'lightning_ir.base.class_factory.BiEncoderBertTokenizerFast'>
            >>> Loading using base class and backbone checkpoint
            >>> type(LightningIRTokenizer.from_pretrained("bert-base-uncased", config=BiEncoderConfig()))
            ...
            <class 'lightning_ir.base.class_factory.BiEncoderBertTokenizerFast'>

        Args:
            model_name_or_path (str): Name or path of the pretrained tokenizer.
        Returns:
            Self: A derived LightningIRTokenizer consisting of a backbone tokenizer and a LightningIRTokenizer mixin.
        Raises:
            ValueError: If called on the abstract class `LightningIRTokenizer` and no config is passed.
        """
        # provides AutoTokenizer.from_pretrained support
        config = kwargs.get("config", None)
        if cls is LightningIRTokenizer or all(issubclass(base, LightningIRTokenizer) for base in cls.__bases__):
            # no backbone models found, create derived lightning-ir tokenizer based on backbone model
            if config is not None:
                ConfigClass = config.__class__
            elif model_name_or_path in CHECKPOINT_MAPPING:
                _config = CHECKPOINT_MAPPING[model_name_or_path]
                ConfigClass = _config.__class__
                if config is None:
                    kwargs["config"] = _config
            elif cls is not LightningIRTokenizer and hasattr(cls, "config_class"):
                ConfigClass = cls.config_class
            else:
                ConfigClass = LightningIRTokenizerClassFactory.get_lightning_ir_config(model_name_or_path)
                if ConfigClass is None:
                    raise ValueError("Pass a config to `from_pretrained`.")
            ConfigClass = getattr(ConfigClass, "mixin_config", ConfigClass)
            backbone_config = LightningIRTokenizerClassFactory.get_backbone_config(model_name_or_path)
            BackboneTokenizers = LightningIRTokenizerClassFactory.get_backbone_tokenizer_classes(backbone_config)
            if kwargs.get("use_fast", True):
                BackboneTokenizer = BackboneTokenizers[1] or BackboneTokenizers[0]
            else:
                BackboneTokenizer = BackboneTokenizers[0] or BackboneTokenizers[1]
            if BackboneTokenizer is None:
                raise ValueError(f"No tokenizer class found for backbone config {type(backbone_config).__name__}.")
            cls = LightningIRTokenizerClassFactory(ConfigClass).from_backbone_class(BackboneTokenizer)
            return cls.from_pretrained(model_name_or_path, *args, **kwargs)
        config = kwargs.pop("config", None)
        if config is not None:
            kwargs.update(config.get_tokenizer_kwargs(cls))
        return super().from_pretrained(model_name_or_path, *args, **kwargs)

    def _save_pretrained(
        self,
        save_directory: str | PathLike,
        file_names: tuple[str],
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
    ) -> tuple[str]:
        # bit of a hack to change the tokenizer class in the stored tokenizer config to only contain the
        # lightning_ir tokenizer class (removing the backbone tokenizer class)
        save_files = super()._save_pretrained(save_directory, file_names, legacy_format, filename_prefix)
        config_file = save_files[0]
        with open(config_file) as file:
            tokenizer_config = json.load(file)

        tokenizer_class = None
        backbone_tokenizer_class = None
        for base in self.__class__.__bases__:
            if issubclass(base, LightningIRTokenizer):
                if tokenizer_class is not None:
                    raise ValueError("Multiple Lightning IR tokenizer classes found.")
                tokenizer_class = base.__name__
                continue
            if issubclass(base, PreTrainedTokenizerBase):
                backbone_tokenizer_class = base.__name__

        tokenizer_config["tokenizer_class"] = tokenizer_class
        tokenizer_config["backbone_tokenizer_class"] = backbone_tokenizer_class

        with open(config_file, "w") as file:
            out_str = json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            file.write(out_str)
        return save_files
