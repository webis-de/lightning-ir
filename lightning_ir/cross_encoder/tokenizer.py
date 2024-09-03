from typing import Dict, List, Sequence, Tuple, Type

from transformers import BatchEncoding

from ..base import LightningIRTokenizer
from .config import CrossEncoderConfig


class CrossEncoderTokenizer(LightningIRTokenizer):

    config_class: Type[CrossEncoderConfig] = CrossEncoderConfig

    def __init__(self, *args, query_length: int = 32, doc_length: int = 512, **kwargs):
        super().__init__(*args, query_length=query_length, doc_length=doc_length, **kwargs)

    def truncate(self, text: Sequence[str], max_length: int) -> List[str]:
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

    def expand_queries(self, queries: Sequence[str], num_docs: Sequence[int]) -> List[str]:
        return [query for query_idx, query in enumerate(queries) for _ in range(num_docs[query_idx])]

    def preprocess(
        self,
        queries: str | Sequence[str] | None,
        docs: str | Sequence[str] | None,
        num_docs: Sequence[int] | None,
    ) -> Tuple[str | Sequence[str], str | Sequence[str]]:
        if queries is None or docs is None:
            raise ValueError("Both queries and docs must be provided.")
        queries_is_string = isinstance(queries, str)
        docs_is_string = isinstance(docs, str)
        if queries_is_string != docs_is_string:
            raise ValueError("Queries and docs must be both lists or both strings.")
        if queries_is_string and docs_is_string:
            queries = [queries]
            docs = [docs]
        truncated_queries = self.truncate(queries, self.query_length)
        truncated_docs = self.truncate(docs, self.doc_length)
        if not queries_is_string:
            if num_docs is None:
                num_docs = [len(docs) // len(queries) for _ in range(len(queries))]
            expanded_queries = self.expand_queries(truncated_queries, num_docs)
            docs = truncated_docs
        else:
            expanded_queries = truncated_queries[0]
            docs = truncated_docs[0]
        return expanded_queries, docs

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        num_docs: Sequence[int] | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        expanded_queries, docs = self.preprocess(queries, docs, num_docs)
        return_tensors = kwargs.get("return_tensors", None)
        if return_tensors is not None:
            kwargs["pad_to_multiple_of"] = 8
        return {"encoding": self(expanded_queries, docs, **kwargs)}
