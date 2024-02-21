from typing import Tuple

import torch
from transformers.models.bert.modeling_bert import BertSelfAttention

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

from .flash_mixin import FlashMixin


class FlashBertMixin(FlashMixin):
    encoder_name = "bert"
    self_attention_pattern = "self"

    @staticmethod
    def flash_attention_forward(
        self: BertSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        key_value_hidden_states = hidden_states
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(key_value_hidden_states))
        value = self.transpose_for_scores(self.value(key_value_hidden_states))

        if (
            flash_attn_func is not None
            and hidden_states.is_cuda
            and (attention_mask is None or not attention_mask.any())
        ):
            context = flash_attn_func(
                query, key, value, self.dropout.p if self.training else 0
            )
        else:
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
