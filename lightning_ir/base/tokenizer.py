from os import PathLike
from typing import Dict, Sequence

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase


class LightningIRTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        tokenizer.init_kwargs.update(kwargs)
        self.__tokenizer = tokenizer

    def __getattr__(self, attr):
        if attr.endswith("__tokenizer"):
            return self.__tokenizer
        return getattr(self.__tokenizer, attr)

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        return self.__tokenizer.__call__(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.__tokenizer)

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        raise NotImplementedError("Tokenizer must implement tokenize method.")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike,
        *init_inputs,
        cache_dir: str | PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
        kwargs.update(tokenizer.init_kwargs)
        return cls(tokenizer, **kwargs)
