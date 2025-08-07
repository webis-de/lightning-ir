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

    def __init__(
        self, *args, query_length: int = 32, doc_length: int = 512, tokenizer_pattern: str | None = None, **kwargs
    ):
        """:class:`.LightningIRTokenizer` for cross-encoder models. Encodes queries and documents jointly and ensures
        that the input sequences are of the correct length.

        Args:
            query_length (int): Maximum number of tokens per query. Defaults to 32.
            doc_length (int): Maximum number of tokens per document. Defaults to 512.
        """
        super().__init__(
            *args, query_length=query_length, doc_length=doc_length, tokenizer_pattern=tokenizer_pattern, **kwargs
        )
        self.tokenizer_pattern = tokenizer_pattern

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

        Args:
            queries (str | Sequence[str] | None): Queries to tokenize. Defaults to None.
            docs (str | Sequence[str] | None): Documents to tokenize. Defaults to None.
            num_docs (Sequence[int] | int | None): Specifies how many documents are passed per query. If a sequence of
                integers, `len(num_docs)` should be equal to the number of queries and `sum(num_docs)` equal to the
                number of documents, i.e., the sequence contains one value per query specifying the number of documents
                for that query. If an integer, assumes an equal number of documents per query. If None, tries to infer
                the number of documents by dividing the number of documents by the number of queries. Defaults to None.
        Returns:
            Dict[str, BatchEncoding]: Tokenized query-document sequence.
        Raises:
            ValueError: If either queries or docs are None.
            ValueError: If queries and docs are not both lists or both strings.
        """
        if queries is None or docs is None:
            raise ValueError("Both queries and docs must be provided.")
        if isinstance(docs, str) and not isinstance(queries, str):
            raise ValueError("Queries and docs must be both lists or both strings.")
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(docs, str):
            docs = [docs]
        num_docs = self._process_num_docs(queries, docs, num_docs)
        queries, docs = self._preprocess(queries, docs, num_docs)

        if self.tokenizer_pattern is not None:
            input_texts = [self.tokenizer_pattern.format(query=query, doc=doc) for query, doc in zip(queries, docs)]
            encoding = self(input_texts, **kwargs)
        else:
            encoding = self(queries, docs, **kwargs)

        return {"encoding": encoding}
