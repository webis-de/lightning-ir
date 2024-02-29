import warnings
from pathlib import Path
from typing import List, Tuple

from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding, BertTokenizerFast


class MVRTokenizer(BertTokenizerFast):
    def __init__(
        self,
        vocab_file: str | Path | None = None,
        tokenizer_file: str | Path | None = None,
        do_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        query_token: str = "[QUE]",
        doc_token: str = "[DOC]",
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
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
            vocab_file,
            tokenizer_file,
            do_lower_case,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            tokenize_chinese_chars,
            strip_accents,
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

    def save_pretrained(
        self,
        save_directory: str | Path,
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        return super().save_pretrained(
            save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs
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

    def __call__(self, *args, internal: bool = False, **kwargs) -> BatchEncoding:
        if not internal:
            warnings.warn(
                "MVRTokenizer is directly called. Use encode_queries or encode_docs "
                "if marker_tokens should be added and query/doc expansion applied."
            )
        return super().__call__(*args, **kwargs)

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
        encoding = self(text, *args, internal=True, **kwargs)
        self._tokenizer.post_processor = orig_post_processor
        return encoding

    def _expand(
        self, encoding: BatchEncoding, attend_to_expanded_tokens: bool
    ) -> BatchEncoding:
        input_ids = encoding["input_ids"]
        input_ids[input_ids == self.pad_token_id] = self.mask_token_id
        encoding["input_ids"] = input_ids
        if attend_to_expanded_tokens:
            encoding["attention_mask"] = None
        return encoding

    def tokenize_queries(
        self, queries: List[str] | str, *args, **kwargs
    ) -> BatchEncoding:
        if self.query_expansion:
            kwargs["max_length"] = self.query_length
            kwargs["padding"] = "max_length"
        encoding = self._encode(
            queries, post_processor=self.query_post_processor, *args, **kwargs
        )
        if self.query_expansion:
            self._expand(encoding, self.attend_to_query_expanded_tokens)
        return encoding

    def tokenize_docs(self, docs: List[str] | str, *args, **kwargs) -> BatchEncoding:
        if self.doc_expansion:
            kwargs["max_length"] = self.doc_length
            kwargs["padding"] = "max_length"
        encoding = self._encode(
            docs, post_processor=self.doc_post_processor, *args, **kwargs
        )
        if self.doc_expansion:
            self._expand(encoding, self.attend_to_doc_expanded_tokens)
        return encoding
