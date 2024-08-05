from typing import Dict, Sequence

from transformers import TOKENIZER_MAPPING, BatchEncoding

from .class_factory import LightningIRTokenizerClassFactory
from .config import LightningIRConfig


class LightningIRTokenizer:

    config_class = LightningIRConfig

    def __init__(self, *args, query_length: int = 32, doc_length: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_length = query_length
        self.doc_length = doc_length

    def tokenize(
        self, queries: str | Sequence[str] | None = None, docs: str | Sequence[str] | None = None, **kwargs
    ) -> Dict[str, BatchEncoding]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args, **kwargs) -> "LightningIRTokenizer":
        config = kwargs.get("config", None)
        if config is not None:
            kwargs.update(config.to_tokenizer_dict())
        if all(issubclass(base, LightningIRTokenizer) for base in cls.__bases__) or cls is LightningIRTokenizer:
            # no backbone models found, create derived lightning-ir tokenizer based on backbone model
            BackboneConfig = LightningIRTokenizerClassFactory.get_backbone_config(model_name_or_path)
            BackboneTokenizers = TOKENIZER_MAPPING[BackboneConfig]
            if kwargs.get("use_fast", True):
                BackboneTokenizer = BackboneTokenizers[1]
            else:
                BackboneTokenizer = BackboneTokenizers[0]
            if config is not None:
                Config = config.__class__
            elif cls is not LightningIRTokenizer and hasattr(cls, "config_class"):
                Config = cls.config_class
            else:
                raise ValueError("Pass a config to `from_pretrained`.")
            cls = LightningIRTokenizerClassFactory(Config).from_backbone_class(BackboneTokenizer)
        return super(LightningIRTokenizer, cls).from_pretrained(model_name_or_path, *args, **kwargs)
