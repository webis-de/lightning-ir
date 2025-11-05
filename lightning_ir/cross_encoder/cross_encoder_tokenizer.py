"""
Tokenizer module for cross-encoder models.

This module contains the tokenizer class cross-encoder models.
"""

from typing import Dict, List, Sequence, Tuple, Type

from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding

from ..base import LightningIRTokenizer, LightningIRTokenizerClassFactory
from .cross_encoder_config import CrossEncoderConfig

SCORING_STRATEGY_POST_PROCESSOR_MAPPING = {
    "t5": {
        "mono": {
            "pattern": "pre que col $A doc col $B rel1 rel2 col eos",
            "special_tokens": [
                ("pre", "▁"),
                ("que", "Query"),
                ("col", ":"),
                ("doc", "▁Document"),
                ("rel1", "▁Relevan"),
                ("rel2", "t"),
                ("eos", "</s>"),
            ],
        },
        "rank": {
            "pattern": "pre que col $A doc col $B eos",
            "special_tokens": [
                ("pre", "▁"),
                ("que", "Query"),
                ("col", ":"),
                ("doc", "▁Document"),
                ("rel1", "▁Relevan"),
                ("rel2", "t"),
                ("eos", "</s>"),
            ],
        },
    },
}


class CrossEncoderTokenizer(LightningIRTokenizer):

    config_class: Type[CrossEncoderConfig] = CrossEncoderConfig
    """Configuration class for the tokenizer."""

    def __init__(
        self,
        *args,
        query_length: int | None = 32,
        doc_length: int | None = 512,
        scoring_strategy: str | None = None,
        **kwargs,
    ):
        """:class:`.LightningIRTokenizer` for cross-encoder models. Encodes queries and documents jointly and ensures
        that the input sequences are of the correct length.

        Args:
            query_length (int | None): Maximum number of tokens per query. If None does not truncate. Defaults to 32.
            doc_length (int | None): Maximum number of tokens per document. If None does not truncate. Defaults to 512.
        """
        super().__init__(
            *args, query_length=query_length, doc_length=doc_length, scoring_strategy=scoring_strategy, **kwargs
        )
        self.scoring_strategy = scoring_strategy
        backbone_model_type = LightningIRTokenizerClassFactory.get_backbone_model_type(self.name_or_path)
        self.post_processor: TemplateProcessing | None = None
        if backbone_model_type in SCORING_STRATEGY_POST_PROCESSOR_MAPPING:
            mapping = SCORING_STRATEGY_POST_PROCESSOR_MAPPING[backbone_model_type]
            if scoring_strategy is not None and scoring_strategy in mapping:
                pattern = mapping[scoring_strategy]["pattern"]
                special_tokens = [
                    (placeholder, self.convert_tokens_to_ids(token))
                    for (placeholder, token) in mapping[scoring_strategy]["special_tokens"]
                ]
                self.post_processor = TemplateProcessing(
                    single=None,
                    pair=pattern,
                    special_tokens=special_tokens,
                )

    def _truncate(self, text: Sequence[str], max_length: int | None) -> List[str]:
        """Encodes a list of texts, truncates them to a maximum number of tokens and decodes them to strings."""
        if max_length is None:
            return text
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

        orig_post_processor = self._tokenizer.post_processor
        if self.post_processor is not None:
            self._tokenizer.post_processor = self.post_processor

        encoding = self(queries, docs, **kwargs)

        if self.post_processor is not None:
            self._tokenizer.post_processor = orig_post_processor

        return {"encoding": encoding}
