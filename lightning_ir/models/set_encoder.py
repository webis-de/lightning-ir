"""
Configuration and model implementation for SetEncoder type models. Originally proposed in
`Set-Encoder: Permutation-Invariant Inter-passage Attention for Listwise Passage Re-ranking with Cross-Encoders
<https://link.springer.com/chapter/10.1007/978-3-031-88711-6_1>`_.
"""

from functools import partial
from typing import Dict, Sequence, Tuple

import torch
from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding

from ..cross_encoder import CrossEncoderOutput, CrossEncoderTokenizer
from .mono import MonoConfig, MonoModel


class SetEncoderConfig(MonoConfig):
    """Configuration class for a SetEncoder model."""

    model_type = "set-encoder"
    """Model type for a SetEncoder model."""

    def __init__(
        self,
        *args,
        depth: int = 100,
        add_extra_token: bool = False,
        sample_missing_docs: bool = True,
        **kwargs,
    ):
        """
        A SetEncoder model encodes a query and a set of documents jointly.
        Each document's embedding is updated with context from the entire set,
        and a relevance score is computed per document using a linear layer.

        Args:
            depth (int): Number of documents to encode per query. Defaults to 100.
            add_extra_token (bool): Whether to add an extra token to the input sequence to separate
                the query from the documents. Defaults to False.
            sample_missing_docs (bool): Whether to sample missing documents when the number of documents is less
                than the specified depth. Defaults to True.
        """

        super().__init__(*args, **kwargs)
        self.depth = depth
        self.add_extra_token = add_extra_token
        self.sample_missing_docs = sample_missing_docs


class SetEncoderModel(MonoModel):
    """SetEncoder model. See :class:`SetEncoderConfig` for configuration options."""

    config_class = SetEncoderConfig
    self_attention_pattern = "self"

    ALLOW_SUB_BATCHING = False  # listwise model

    def __init__(self, config: SetEncoderConfig, *args, **kwargs):
        """Initializes a SetEncoder model give a :class:`SetEncoderConfig`.

        Args:
            config (SetEncoderConfig): Configuration for the SetEncoder model.
        """
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
        """
        Extends the attention mask to account for the number of documents per query.

        Args:
            attention_mask (torch.Tensor): Attention mask for the input sequence.
            input_shape (Tuple[int, ...]): Shape of the input sequence.
            device (torch.device | None): Device to move the attention mask to. Defaults to None.
            dtype (torch.dtype | None): Data type of the attention mask. Defaults to None.
            num_docs (Sequence[int] | None): Specifies how many documents are passed per query. If a sequence of
                integers, `len(num_doc)` should be equal to the number of queries and `sum(num_docs)` equal to the
                number of documents, i.e., the sequence contains one value per query specifying the number of documents
                for that query. If an integer, assumes an equal number of documents per query. If None, tries to infer
                the number of documents by dividing the number of documents by the number of queries. Defaults to None.
            Returns:
                torch.Tensor: Extended attention mask.
        """
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
        """Computes contextualized embeddings for the joint query-document input sequence and computes a relevance
        score.

        Args:
            encoding (BatchEncoding): Tokenizer encoding for the joint query-document input sequence.
        Returns:
            CrossEncoderOutput: Output of the model.
        """
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
        """Performs the attention forward pass for the SetEncoder model.

        Args:
            _self (SetEncoderModel): Reference to the SetEncoder instance.
            self (torch.nn.Module): Reference to the attention module.
            hidden_states (torch.Tensor): Hidden states from the previous layer.
            attention_mask (torch.FloatTensor | None): Attention mask for the input sequence.
            num_docs (Sequence[int]): Specifies how many documents are passed per query. If a sequence of integers,
                `len(num_doc)` should be equal to the number of queries and `sum(num_docs)` equal to the number of
                documents, i.e., the sequence contains one value per query specifying the number of documents
                for that query. If an integer, assumes an equal number of documents per query. If None, tries to infer
                the number of documents by dividing the number of documents by the number of queries.
        Returns:
            Tuple[torch.Tensor]: Contextualized embeddings.
        """
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
        """Concatenates the hidden states of other documents to the hidden states of the query and documents.

        Args:
            hidden_states (torch.Tensor): Hidden states of the query and documents.
            num_docs (Sequence[int]): Specifies how many documents are passed per query. If a sequence of integers,
                `len(num_doc)` should be equal to the number of queries and `sum(num_docs)` equal to the number of
                documents, i.e., the sequence contains one value per query specifying the number of documents
                for that query. If an integer, assumes an equal number of documents per query. If None, tries to infer
                the number of documents by dividing the number of documents by the number of queries.
        Returns:
            torch.Tensor: Concatenated hidden states of the query and documents.
        """
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
        """Initializes a SetEncoder tokenizer.

        Args:
            query_length (int): Maximum query length. Defaults to 32.
            doc_length (int): Maximum document length. Defaults to 512.
            add_extra_token (bool): Whether to add an extra interaction token. Defaults to False.
        """
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
                ValueError: If both queries and docs are None.
                ValueError: If queries and docs are not both lists or both strings.
        """
        encoding_dict = super().tokenize(queries, docs, num_docs, **kwargs)
        encoding_dict["encoding"] = BatchEncoding({**encoding_dict["encoding"], "num_docs": num_docs})
        return encoding_dict
