from functools import partial
from typing import Dict, Sequence, Tuple

import torch
from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding

from ..cross_encoder.cross_encoder_config import CrossEncoderConfig
from ..cross_encoder.cross_encoder_model import CrossEncoderModel, CrossEncoderOutput
from ..cross_encoder.cross_encoder_tokenizer import CrossEncoderTokenizer


class SetEncoderConfig(CrossEncoderConfig):
    model_type = "set-encoder"

    def __init__(
        self,
        *args,
        depth: int = 100,
        add_extra_token: bool = False,
        sample_missing_docs: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.add_extra_token = add_extra_token
        self.sample_missing_docs = sample_missing_docs


class SetEncoderModel(CrossEncoderModel):
    config_class = SetEncoderConfig
    self_attention_pattern = "self"

    ALLOW_SUB_BATCHING = False  # listwise model

    def __init__(self, config: SetEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: SetEncoderConfig
        self.attn_implementation = "eager"
        if self.config.backbone_model_type is not None and self.config.backbone_model_type not in ("bert", "electra"):
            raise ValueError(
                f"SetEncoderModel does not support backbone model type {self.config.backbone_model_type}. "
                f"Supported types are 'bert' and 'electra'."
            )

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_docs: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if num_docs is not None:
            eye = (1 - torch.eye(self.config.depth, device=device)).long()
            if not self.config.sample_missing_docs:
                eye = eye[:, : max(num_docs)]
            other_doc_attention_mask = torch.cat([eye[:n] for n in num_docs])
            attention_mask = torch.cat(
                [attention_mask, other_doc_attention_mask.to(attention_mask)],
                dim=-1,
            )
            input_shape = tuple(attention_mask.shape)
        return super().get_extended_attention_mask(attention_mask, input_shape, device, dtype)

    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        num_docs = encoding.pop("num_docs", None)
        self.get_extended_attention_mask = partial(self.get_extended_attention_mask, num_docs=num_docs)
        for name, module in self.named_modules():
            if name.endswith(self.self_attention_pattern):
                module.forward = partial(self.attention_forward, self, module, num_docs=num_docs)
        return super().forward(encoding)

    @staticmethod
    def attention_forward(
        _self,
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None,
        *args,
        num_docs: Sequence[int],
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        key_value_hidden_states = hidden_states
        if num_docs is not None:
            key_value_hidden_states = _self.cat_other_doc_hidden_states(hidden_states, num_docs)
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(key_value_hidden_states))
        value = self.transpose_for_scores(self.value(key_value_hidden_states))

        context = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask.to(query.dtype) if attention_mask is not None else None,
            self.dropout.p if self.training else 0,
        )

        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(new_context_shape)
        return (context,)

    def cat_other_doc_hidden_states(
        self,
        hidden_states: torch.Tensor,
        num_docs: Sequence[int],
    ) -> torch.Tensor:
        idx = 1 if self.config.add_extra_token else 0
        split_other_doc_hidden_states = torch.split(hidden_states[:, idx], list(num_docs))
        repeated_other_doc_hidden_states = []
        for idx, h_states in enumerate(split_other_doc_hidden_states):
            missing_docs = 0 if self.config.depth is None else self.config.depth - num_docs[idx]
            if missing_docs and self.config.sample_missing_docs:
                mean = h_states.mean(0, keepdim=True).expand(missing_docs, -1)
                if num_docs[idx] == 1:
                    std = torch.zeros_like(mean)
                else:
                    std = h_states.std(0, keepdim=True).expand(missing_docs, -1)
                sampled_h_states = torch.normal(mean, std).to(h_states)
                h_states = torch.cat([h_states, sampled_h_states])
            repeated_other_doc_hidden_states.append(h_states.unsqueeze(0).expand(num_docs[idx], -1, -1))
        other_doc_hidden_states = torch.cat(repeated_other_doc_hidden_states)
        key_value_hidden_states = torch.cat([hidden_states, other_doc_hidden_states], dim=1)
        return key_value_hidden_states


class SetEncoderTokenizer(CrossEncoderTokenizer):

    config_class = SetEncoderConfig
    """Configuration class for the tokenizer."""

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        add_extra_token: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args, query_length=query_length, doc_length=doc_length, add_extra_token=add_extra_token, **kwargs
        )
        self.add_extra_token = add_extra_token
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
