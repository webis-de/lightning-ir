"""
Tokenizer module for cross-encoder models.

This module contains the tokenizer class cross-encoder models.
"""

from typing import Dict, List, Sequence, Tuple, Type

from transformers import BatchEncoding

from ..base import LightningIRTokenizer
from .cross_encoder_config import CrossEncoderConfig


class CrossEncoderTokenizer(LightningIRTokenizer):

    config_class: Type[CrossEncoderConfig] = CrossEncoderConfig
    """Configuration class for the tokenizer."""

    def __init__(self, *args, query_length: int = 32, doc_length: int = 512, **kwargs):
        """:class:`.LightningIRTokenizer` for cross-encoder models. Encodes queries and documents jointly and ensures
        that the input sequences are of the correct length.

        :param query_length: Maximum number of tokens per query, defaults to 32
        :type query_length: int, optional
        :param doc_length: Maximum number of tokens per document, defaults to 512
        :type doc_length: int, optional
        :type doc_length: int, optional
        """
        super().__init__(*args, query_length=query_length, doc_length=doc_length, **kwargs)

    def _truncate(self, text: Sequence[str], max_length: int) -> List[str]:
        """Encodes a list of texts, truncates them to a maximum number of tokens and decodes them to strings."""
        return self.batch_decode(
            self(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_attention_mask=False,
                return_token_type_ids=False,
            ).input_ids
        )

    def _repeat_queries(self, queries: Sequence[str], num_docs: Sequence[int]) -> List[str]:
        """Repeats queries to match the number of documents."""
        return [query for query_idx, query in enumerate(queries) for _ in range(num_docs[query_idx])]

    def _preprocess(
        self,
        queries: Sequence[str],
        docs: Sequence[str],
        num_docs: Sequence[int],
    ) -> Tuple[str | Sequence[str], str | Sequence[str]]:
        """Preprocesses queries and documents to ensure that they are truncated their respective maximum lengths."""
        truncated_queries = self._repeat_queries(self._truncate(queries, self.query_length), num_docs)
        truncated_docs = self._truncate(docs, self.doc_length)
        return truncated_queries, truncated_docs

    def _process_num_docs(
        self,
        queries: str | Sequence[str],
        docs: str | Sequence[str],
        num_docs: Sequence[int] | int | None,
    ) -> List[int]:
        if num_docs is None:
            if isinstance(num_docs, int):
                num_docs = [num_docs] * len(queries)
            else:
                if len(docs) % len(queries) != 0:
                    raise ValueError("Number of documents must be divisible by the number of queries.")
                num_docs = [len(docs) // len(queries) for _ in range(len(queries))]
        return num_docs

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        num_docs: Sequence[int] | int | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        """Tokenizes queries and documents into a single sequence of tokens.

        :param queries: Queries to tokenize, defaults to None
        :type queries: str | Sequence[str] | None, optional
        :param docs: Documents to tokenize, defaults to None
        :type docs: str | Sequence[str] | None, optional
        :param num_docs: Specifies how many documents are passed per query. If a sequence of integers, `len(num_doc)`
            should be equal to the number of queries and `sum(num_docs)` equal to the number of documents, i.e., the
            sequence contains one value per query specifying the number of documents for that query. If an integer,
            assumes an equal number of documents per query. If None, tries to infer the number of documents by dividing
            the number of documents by the number of queries, defaults to None
        :type num_docs: Sequence[int] | int | None, optional
        :return: Tokenized query-document sequence
        :rtype: Dict[str, BatchEncoding]
        """
        if queries is None or docs is None:
            raise ValueError("Both queries and docs must be provided.")
        if isinstance(docs, str) and not isinstance(queries, str):
            raise ValueError("Queries and docs must be both lists or both strings.")
        is_string_queries = False
        is_string_docs = False
        if isinstance(queries, str):
            queries = [queries]
            is_string_queries = True
        if isinstance(docs, str):
            docs = [docs]
            is_string_docs = True
        is_string_both = is_string_queries and is_string_docs
        num_docs = self._process_num_docs(queries, docs, num_docs)
        queries, docs = self._preprocess(queries, docs, num_docs)
        return_tensors = kwargs.get("return_tensors", None)
        if return_tensors is not None:
            kwargs["pad_to_multiple_of"] = 8
        if is_string_both:
            encoding = self(queries[0], docs[0], **kwargs)
        else:
            encoding = self(queries, docs, **kwargs)
        return {"encoding": encoding}
