"""
Tokenizer module for bi-encoder models.

This module contains the tokenizer class bi-encoder models.
"""

import warnings
from typing import Dict, Literal, Sequence, Type

from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding

from ..base import LightningIRClassFactory, LightningIRTokenizer
from .bi_encoder_config import BiEncoderConfig

ADD_MARKER_TOKEN_MAPPING = {
    "bert": {
        "single": "[CLS] {TOKEN} $0 [SEP]",
        "pair": "[CLS] {TOKEN_1} $A [SEP] {TOKEN_2} $B:1 [SEP]:1",
    },
    "modernbert": {
        "single": "[CLS] {TOKEN} $0 [SEP]",
        "pair": "[CLS] {TOKEN_1} $A [SEP] {TOKEN_2} $B:1 [SEP]:1",
    },
}


class BiEncoderTokenizer(LightningIRTokenizer):

    config_class: Type[BiEncoderConfig] = BiEncoderConfig
    """Configuration class for the tokenizer."""

    QUERY_TOKEN: str = "[QUE]"
    """Token to mark a query sequence."""
    DOC_TOKEN: str = "[DOC]"
    """Token to mark a document sequence."""

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        add_marker_tokens: bool = False,
        **kwargs,
    ):
        """:class:`.LightningIRTokenizer` for bi-encoder models. Encodes queries and documents separately. Optionally
        adds marker tokens are added to encoded input sequences.

        Args:
            query_length (int): Maximum query length in number of tokens. Defaults to 32.
            doc_length (int): Maximum document length in number of tokens. Defaults to 512.
            add_marker_tokens (bool): Whether to add marker tokens to the query and document input sequences.
                Defaults to False.
        Raises:
            ValueError: If `add_marker_tokens` is True and a non-supported tokenizer is used.
        """
        super().__init__(
            *args,
            query_length=query_length,
            doc_length=doc_length,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
        self.query_length = query_length
        self.doc_length = doc_length
        self.add_marker_tokens = add_marker_tokens

        self.query_post_processor: TemplateProcessing | None = None
        self.doc_post_processor: TemplateProcessing | None = None
        if add_marker_tokens:
            backbone_model_type = LightningIRClassFactory.get_backbone_model_type(self.name_or_path)
            if backbone_model_type not in ADD_MARKER_TOKEN_MAPPING:
                raise ValueError(
                    f"Adding marker tokens is not supported for the backbone model type '{backbone_model_type}'. "
                    f"Supported types are: [{', '.join(ADD_MARKER_TOKEN_MAPPING.keys())}]. "
                    "Please set `add_marker_tokens=False` "
                    "or add the backbone model type to `ADD_MARKER_TOKEN_MAPPING`."
                )
            self.add_tokens([self.QUERY_TOKEN, self.DOC_TOKEN], special_tokens=True)
            pattern = ADD_MARKER_TOKEN_MAPPING[backbone_model_type]
            self.query_post_processor = TemplateProcessing(
                single=pattern["single"].format(TOKEN=self.QUERY_TOKEN),
                pair=pattern["pair"].format(TOKEN_1=self.QUERY_TOKEN, TOKEN_2=self.DOC_TOKEN),
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    (self.QUERY_TOKEN, self.query_token_id),
                    (self.DOC_TOKEN, self.doc_token_id),
                ],
            )
            self.doc_post_processor = TemplateProcessing(
                single=pattern["single"].format(TOKEN=self.DOC_TOKEN),
                pair=pattern["pair"].format(TOKEN_1=self.QUERY_TOKEN, TOKEN_2=self.DOC_TOKEN),
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    (self.QUERY_TOKEN, self.query_token_id),
                    (self.DOC_TOKEN, self.doc_token_id),
                ],
            )

    @property
    def query_token_id(self) -> int | None:
        """The token id of the query token if marker tokens are added.

        Return:
            Token id of the query token if added, otherwise None.
        """
        if self.QUERY_TOKEN in self.added_tokens_encoder:
            return self.added_tokens_encoder[self.QUERY_TOKEN]
        return None

    @property
    def doc_token_id(self) -> int | None:
        """The token id of the document token if marker tokens are added.

        Returns:
            Token id of the document token if added, otherwise None.
        """
        if self.DOC_TOKEN in self.added_tokens_encoder:
            return self.added_tokens_encoder[self.DOC_TOKEN]
        return None

    def __call__(self, *args, warn: bool = True, **kwargs) -> BatchEncoding:
        """Overrides the PretrainedTokenizer.__call___ method to warn the user to use :meth:`.tokenize_query` and
        :meth:`.tokenize_doc` methods instead.

        .. PretrainedTokenizer.__call__: \
https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__

        Args:
            text (str | Sequence[str]): Text to tokenize.
            warn (bool): Set to False to silence warning. Defaults to True.
        Returns:
            BatchEncoding: Tokenized text.
        """
        if warn:
            warnings.warn(
                "BiEncoderTokenizer is being directly called. Use `tokenize`, `tokenize_query`, or `tokenize_doc` "
                "to make sure tokenization is done correctly.",
            )
        return super().__call__(*args, **kwargs)

    def _encode(
        self,
        text: str | Sequence[str],
        *args,
        post_processor: TemplateProcessing | None = None,
        **kwargs,
    ) -> BatchEncoding:
        """Encodes text with an optional post-processor."""
        orig_post_processor = self._tokenizer.post_processor
        if post_processor is not None:
            self._tokenizer.post_processor = post_processor
        if kwargs.get("return_tensors", None) is not None:
            kwargs["pad_to_multiple_of"] = 8
        encoding = self(text, *args, warn=False, **kwargs)
        self._tokenizer.post_processor = orig_post_processor
        return encoding

    def _expand(self, encoding: BatchEncoding, attend_to_expanded_tokens: bool) -> BatchEncoding:
        """Applies mask expansion to the input encoding."""
        input_ids = encoding["input_ids"]
        input_ids[input_ids == self.pad_token_id] = self.mask_token_id
        encoding["input_ids"] = input_ids
        if attend_to_expanded_tokens:
            encoding["attention_mask"].fill_(1)
        return encoding

    def tokenize_input_sequence(
        self, text: Sequence[str] | str, input_type: Literal["query", "doc"], *args, **kwargs
    ) -> BatchEncoding:
        """Tokenizes an input sequence. This method is used to tokenize both queries and documents.

        Args:
            text (Sequence[str] | str): Input text to tokenize.
            input_type (Literal["query", "doc"]): Type of input, either "query" or "doc".
        Returns:
            BatchEncoding: Tokenized input sequences.
        """
        post_processer = getattr(self, f"{input_type}_post_processor")
        kwargs["max_length"] = getattr(self, f"{input_type}_length")
        if "padding" not in kwargs:
            kwargs["truncation"] = True
        return self._encode(text, *args, post_processor=post_processer, **kwargs)

    def tokenize_query(self, queries: Sequence[str] | str, *args, **kwargs) -> BatchEncoding:
        """Tokenizes input queries.

        Args:
            queries (Sequence[str] | str): Query or queries to tokenize.
        Returns:
            BatchEncoding: Tokenized queries.
        """
        encoding = self.tokenize_input_sequence(queries, "query", *args, **kwargs)
        return encoding

    def tokenize_doc(self, docs: Sequence[str] | str, *args, **kwargs) -> BatchEncoding:
        """Tokenizes input documents.

        Args:
            docs (Sequence[str] | str): Document or documents to tokenize.
        Returns:
            BatchEncoding: Tokenized documents.
        """
        encoding = self.tokenize_input_sequence(docs, "doc", *args, **kwargs)
        return encoding

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        """Tokenizes queries and documents.

        Args:
            queries (str | Sequence[str] | None): Queries to tokenize. Defaults to None.
            docs (str | Sequence[str] | None): Documents to tokenize. Defaults to None.
        Returns:
            Dict[str, BatchEncoding]: Dictionary containing tokenized queries and documents.
        """
        encodings = {}
        kwargs.pop("num_docs", None)
        if queries is not None:
            encodings["query_encoding"] = self.tokenize_query(queries, **kwargs)
        if docs is not None:
            encodings["doc_encoding"] = self.tokenize_doc(docs, **kwargs)
        return encodings
