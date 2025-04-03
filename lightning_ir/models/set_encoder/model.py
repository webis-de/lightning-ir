from functools import partial
from typing import Sequence, Tuple

import torch
from transformers import BatchEncoding

from ...cross_encoder.model import CrossEncoderModel, CrossEncoderOutput
from .config import SetEncoderConfig


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
