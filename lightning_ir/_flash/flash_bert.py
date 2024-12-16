from typing import Tuple

import torch
from transformers.models.bert.modeling_bert import BertSelfAttention

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def flash_attention_forward(
    self: BertSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.FloatTensor | None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor]:
    query = self.transpose_for_scores(self.query(hidden_states))
    key = self.transpose_for_scores(self.key(hidden_states))
    value = self.transpose_for_scores(self.value(hidden_states))

    if attention_mask is not None and not attention_mask.any():
        attention_mask = None

    if flash_attn_func is not None and hidden_states.is_cuda and attention_mask is None:
        context = (
            flash_attn_func(
                query.bfloat16().transpose(1, 2),
                key.bfloat16().transpose(1, 2),
                value.bfloat16().transpose(1, 2),
                self.dropout.p if self.training else 0,
            )
            .transpose(1, 2)
            .to(query.dtype)
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


SELF_ATTENTION_PATTERN = "self"
