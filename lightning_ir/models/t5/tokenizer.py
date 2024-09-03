from typing import Dict, Literal, Sequence, Type

from transformers import BatchEncoding

from ...cross_encoder.tokenizer import CrossEncoderTokenizer
from .config import T5CrossEncoderConfig


class T5CrossEncoderTokenizer(CrossEncoderTokenizer):

    config_class: Type[T5CrossEncoderConfig] = T5CrossEncoderConfig

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        decoder_strategy: Literal["mono", "rank"] = "mono",
        **kwargs,
    ):
        super().__init__(
            *args, query_length=query_length, doc_length=doc_length, decoder_strategy=decoder_strategy, **kwargs
        )
        self.decoder_strategy = decoder_strategy

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        num_docs: Sequence[int] | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        expanded_queries, docs = self.preprocess(queries, docs, num_docs)
        if self.decoder_strategy == "mono":
            pattern = "Query: {query} Document: {doc} Relevant:"
        elif self.decoder_strategy == "rank":
            pattern = "Query: {query} Document: {doc}"
        else:
            raise ValueError(f"Unknown decoder strategy: {self.decoder_strategy}")
        input_texts = [pattern.format(query=query, doc=doc) for query, doc in zip(expanded_queries, docs)]

        return_tensors = kwargs.get("return_tensors", None)
        if return_tensors is not None:
            kwargs["pad_to_multiple_of"] = 8
        return {"encoding": self(input_texts, **kwargs)}
