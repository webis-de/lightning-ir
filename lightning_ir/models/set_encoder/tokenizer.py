from typing import Dict, Sequence

from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding

from ...cross_encoder.tokenizer import CrossEncoderTokenizer


class SetEncoderTokenizer(CrossEncoderTokenizer):

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        add_extra_token: bool = False,
        **kwargs,
    ):
        super().__init__(*args, query_length=query_length, doc_length=doc_length, **kwargs)
        self.interaction_token = "[INT]"
        if add_extra_token:
            self.add_tokens([self.interaction_token], special_tokens=True)
            self._tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $0 [SEP]",
                pair="[CLS] [INT] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    ("[INT]", self.interaction_token_id),
                ],
            )

    @property
    def interaction_token_id(self) -> int:
        if self.interaction_token in self.added_tokens_encoder:
            return self.added_tokens_encoder[self.interaction_token]
        raise ValueError(f"Token {self.interaction_token} not found in tokenizer")

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
        return {"encoding": BatchEncoding({**encoding, "num_docs": num_docs})}
