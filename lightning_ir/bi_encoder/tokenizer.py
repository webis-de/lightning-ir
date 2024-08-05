import warnings
from typing import Dict, Sequence

from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding, BertTokenizer, BertTokenizerFast

from ..base import LightningIRTokenizer
from .config import BiEncoderConfig


class BiEncoderTokenizer(LightningIRTokenizer):

    config_class = BiEncoderConfig

    def __init__(
        self,
        *args,
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
            *args,
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

        self.query_post_processor: TemplateProcessing | None = None
        self.doc_post_processor: TemplateProcessing | None = None
        if add_marker_tokens:
            # TODO support other tokenizers
            if not isinstance(self, (BertTokenizer, BertTokenizerFast)):
                raise ValueError("Adding marker tokens is only supported for BertTokenizer.")
            self.add_tokens([query_token, doc_token], special_tokens=True)
            self.query_post_processor = TemplateProcessing(
                single=f"[CLS] {self.query_token} $0 [SEP]",
                pair=f"[CLS] {self.query_token} $A [SEP] {self.doc_token} $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    (self.query_token, self.query_token_id),
                    (self.doc_token, self.doc_token_id),
                ],
            )
            self.doc_post_processor = TemplateProcessing(
                single=f"[CLS] {self.doc_token} $0 [SEP]",
                pair=f"[CLS] {self.query_token} $A [SEP] {self.doc_token} $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    (self.query_token, self.query_token_id),
                    (self.doc_token, self.doc_token_id),
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
                "BiEncoderTokenizer is being directly called. Use tokenize_query and "
                "tokenize_doc to make sure marker_tokens and query/doc expansion is "
                "applied."
            )
        return super().__call__(*args, **kwargs)

    def _encode(
        self,
        text: str | Sequence[str],
        *args,
        post_processor: TemplateProcessing | None = None,
        **kwargs,
    ) -> BatchEncoding:
        orig_post_processor = self._tokenizer.post_processor
        if post_processor is not None:
            self._tokenizer.post_processor = post_processor
        if kwargs.get("return_tensors", None) is not None:
            kwargs["pad_to_multiple_of"] = 8
        encoding = self(text, *args, warn=False, **kwargs)
        self._tokenizer.post_processor = orig_post_processor
        return encoding

    def _expand(self, encoding: BatchEncoding, attend_to_expanded_tokens: bool) -> BatchEncoding:
        input_ids = encoding["input_ids"]
        input_ids[input_ids == self.pad_token_id] = self.mask_token_id
        encoding["input_ids"] = input_ids
        if attend_to_expanded_tokens:
            encoding["attention_mask"].fill_(1)
        return encoding

    def tokenize_query(self, queries: Sequence[str] | str, *args, **kwargs) -> BatchEncoding:
        kwargs["max_length"] = self.query_length
        if self.query_expansion:
            kwargs["padding"] = "max_length"
        else:
            kwargs["truncation"] = True
        encoding = self._encode(queries, *args, post_processor=self.query_post_processor, **kwargs)
        if self.query_expansion:
            self._expand(encoding, self.attend_to_query_expanded_tokens)
        return encoding

    def tokenize_doc(self, docs: Sequence[str] | str, *args, **kwargs) -> BatchEncoding:
        kwargs["max_length"] = self.doc_length
        if self.doc_expansion:
            kwargs["padding"] = "max_length"
        else:
            kwargs["truncation"] = True
        encoding = self._encode(docs, *args, post_processor=self.doc_post_processor, **kwargs)
        if self.doc_expansion:
            self._expand(encoding, self.attend_to_doc_expanded_tokens)
        return encoding

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        encodings = {}
        kwargs.pop("num_docs", None)
        if queries is not None:
            encodings["query_encoding"] = self.tokenize_query(queries, **kwargs)
        if docs is not None:
            encodings["doc_encoding"] = self.tokenize_doc(docs, **kwargs)
        return encodings
