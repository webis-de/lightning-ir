from os import PathLike
import warnings
from typing import Dict, List

from transformers import AutoTokenizer

from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding, PreTrainedTokenizerBase


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
        queries: str | None = None,
        docs: str | None = None,
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


class CrossEncoderTokenizer(LightningIRTokenizer):
    def tokenize(
        self,
        queries: str | None = None,
        docs: str | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        if queries is None or docs is None:
            raise ValueError("Both queries and docs must be provided.")
        num_docs = len(docs) // len(queries)
        expanded_queries = [query for query in queries for _ in range(num_docs)]
        return {"encoding": self(expanded_queries, docs, **kwargs)}


class BiEncoderTokenizer(LightningIRTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query_token: str = "[QUE]",
        doc_token: str = "[DOC]",
        query_expansion: bool = False,
        query_length: int = 32,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        doc_length: int = 512,
        attend_to_doc_expanded_tokens: bool = False,
        add_marker_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            query_token=query_token,
            doc_token=doc_token,
            query_expansion=query_expansion,
            query_length=query_length,
            attend_to_query_expanded_tokens=attend_to_query_expanded_tokens,
            doc_expansion=doc_expansion,
            doc_length=doc_length,
            attend_to_doc_expanded_tokens=attend_to_doc_expanded_tokens,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
        self.query_expansion = query_expansion
        self.query_length = query_length
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.doc_length = doc_length
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens
        self.add_marker_tokens = add_marker_tokens

        self._query_token = query_token
        self._doc_token = doc_token

        self.query_post_processor = None
        self.doc_post_processor = None
        if add_marker_tokens:
            self.add_tokens([query_token, doc_token], special_tokens=True)
            self.query_post_processor = TemplateProcessing(
                single="[CLS] [QUE] $0 [SEP]",
                pair="[CLS] [QUE] $A [SEP] [DOC] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    ("[QUE]", self.query_token_id),
                    ("[DOC]", self.doc_token_id),
                ],
            )
            self.doc_post_processor = TemplateProcessing(
                single="[CLS] [DOC] $0 [SEP]",
                pair="[CLS] [SEP] $A [SEP] [DOC] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    ("[QUE]", self.query_token_id),
                    ("[DOC]", self.doc_token_id),
                ],
            )

    @property
    def query_token(self) -> str:
        return self._query_token

    @property
    def doc_token(self) -> str:
        return self._doc_token

    @property
    def query_token_id(self) -> int | None:
        if self.query_token in self.added_tokens_encoder:
            return self.added_tokens_encoder[self.query_token]
        return None

    @property
    def doc_token_id(self) -> int | None:
        if self.doc_token in self.added_tokens_encoder:
            return self.added_tokens_encoder[self.doc_token]
        return None

    def __call__(self, *args, warn: bool = True, **kwargs) -> BatchEncoding:
        if warn:
            warnings.warn(
                "BiEncoderTokenizer is being directly called. Use tokenize_queries and "
                "tokenize_docs to make sure marker_tokens and query/doc expansion is "
                "applied."
            )
        return self.__tokenizer.__call__(*args, **kwargs)

    def _encode(
        self,
        text: str | List[str],
        *args,
        post_processor: TemplateProcessing | None = None,
        **kwargs,
    ) -> BatchEncoding:
        orig_post_processor = self._tokenizer.post_processor
        if post_processor is not None:
            self._tokenizer.post_processor = post_processor
        encoding = self(text, *args, warn=False, **kwargs)
        self._tokenizer.post_processor = orig_post_processor
        return encoding

    def _expand(
        self, encoding: BatchEncoding, attend_to_expanded_tokens: bool
    ) -> BatchEncoding:
        input_ids = encoding["input_ids"]
        input_ids[input_ids == self.pad_token_id] = self.mask_token_id
        encoding["input_ids"] = input_ids
        if attend_to_expanded_tokens:
            encoding["attention_mask"].fill_(1)
        return encoding

    def tokenize_queries(
        self, queries: List[str] | str, *args, **kwargs
    ) -> BatchEncoding:
        kwargs["max_length"] = self.query_length
        if self.query_expansion:
            kwargs["padding"] = "max_length"
        else:
            kwargs["truncation"] = True
        encoding = self._encode(
            queries, *args, post_processor=self.query_post_processor, **kwargs
        )
        if self.query_expansion:
            self._expand(encoding, self.attend_to_query_expanded_tokens)
        return encoding

    def tokenize_docs(self, docs: List[str] | str, *args, **kwargs) -> BatchEncoding:
        kwargs["max_length"] = self.doc_length
        if self.doc_expansion:
            kwargs["padding"] = "max_length"
        else:
            kwargs["truncation"] = True
        encoding = self._encode(
            docs, *args, post_processor=self.doc_post_processor, **kwargs
        )
        if self.doc_expansion:
            self._expand(encoding, self.attend_to_doc_expanded_tokens)
        return encoding

    def tokenize(
        self,
        queries: str | None = None,
        docs: str | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        encodings = {}
        if queries is not None:
            encodings["query_encoding"] = self.tokenize_queries(queries, **kwargs)
        if docs is not None:
            encodings["doc_encoding"] = self.tokenize_docs(docs, **kwargs)
        return encodings
