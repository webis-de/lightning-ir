from typing import Dict, Sequence

from transformers import BatchEncoding

from ..base import LightningIRTokenizer
from .config import CrossEncoderConfig


class CrossEncoderTokenizer(LightningIRTokenizer):

    config_class = CrossEncoderConfig

    def __init__(self, *args, query_length: int = 32, doc_length: int = 512, **kwargs):
        super().__init__(*args, query_length=query_length, doc_length=doc_length, **kwargs)

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        num_docs: Sequence[int] | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        if queries is None or docs is None:
            raise ValueError("Both queries and docs must be provided.")
        queries_is_list = isinstance(queries, list)
        docs_is_list = isinstance(docs, list)
        if queries_is_list != docs_is_list:
            raise ValueError("Queries and docs must be both lists or both strings.")
        if not queries_is_list:
            queries = [queries]
            docs = [docs]
        truncated_queries = self.batch_decode(
            self(
                queries,
                add_special_tokens=False,
                truncation=True,
                max_length=self.query_length,
                return_attention_mask=False,
                return_token_type_ids=False,
            ).input_ids
        )
        truncated_docs = self.batch_decode(
            self(
                docs,
                add_special_tokens=False,
                truncation=True,
                max_length=self.doc_length,
                return_attention_mask=False,
                return_token_type_ids=False,
            ).input_ids
        )
        if queries_is_list:
            if num_docs is None:
                num_docs = [len(docs) // len(queries) for _ in range(len(queries))]
            expanded_queries = [
                truncated_query
                for query_idx, truncated_query in enumerate(truncated_queries)
                for _ in range(num_docs[query_idx])
            ]
            docs = truncated_docs
        else:
            expanded_queries = truncated_queries[0]
            docs = truncated_docs[0]
        return_tensors = kwargs.get("return_tensors", None)

        if return_tensors is not None:
            kwargs["pad_to_multiple_of"] = 8
        return {"encoding": self(expanded_queries, docs, **kwargs)}
