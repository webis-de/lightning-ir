import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import scaled_dot_product_attention

from typing import Optional
import numpy as np

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    DataCollatorForLanguageModeling,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

from .rotary import compute_cos_sin, apply_rotary_emb


# RoPE base (theta); matches the original default of 10000.0.
ROPE_THETA = 10000.0


class SwiGLU(nn.Module):
    """Pure-torch SwiGLU. The fused ``w12`` (gate+value) / ``w3`` layout keeps the original
    checkpoint's weight names, so weights load unchanged. Computes ``w3(silu(x1) * x2)`` with
    ``x1, x2 = w12(x).chunk(2, -1)``."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class DataCollatorWithPacking(DataCollatorForLanguageModeling):
    def __init__(self, pack_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.pack_sequences = pack_sequences

    def __call__(self, batch):
        if self.pack_sequences:
            if "position_ids" not in batch[0]:
                for item in batch:
                    item["position_ids"] = list(range(len(item["input_ids"])))

            input_ids_list = [item["input_ids"] for item in batch]
            position_ids_list = [item["position_ids"] for item in batch]
            seqlens = np.array([0] + [len(ids) for ids in input_ids_list])

            packed_batch = {
                "position_ids": np.concatenate(position_ids_list, axis=0),
                "input_ids": np.concatenate(input_ids_list, axis=0),
                "cu_seqlens": np.cumsum(seqlens),
                "max_seqlen": max(seqlens),
            }

            batch = super().__call__([packed_batch])
            batch["cu_seqlens"] = batch["cu_seqlens"].to(torch.int32).squeeze()
        else:
            batch = super().__call__(batch)
            batch["attention_mask"] = batch["attention_mask"].to(torch.bool)

        return batch


class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        norm_eps: float = 1e-06,
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        max_length: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.kwargs = kwargs


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)

        # Feedforward network
        multiple_of = 8
        intermediate_size = int(2 * config.intermediate_size / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)

        # Layer norms
        self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        output_attentions: bool,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
    ):
        # Attention
        attn_output, attn_weights = self._att_block(
            self.attention_norm(x), attention_mask, cos, sin, output_attentions, max_seqlen, cu_seqlens
        )

        # Residual
        x = x + attn_output

        # Feed-forward
        x = x + self.ffn(self.ffn_norm(x))

        return x, attn_weights

    def _att_block(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        output_attentions: bool,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = (
            self.qkv(x)
            .view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3)
            .chunk(3, axis=-1)
        )

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # Attn block
        attn_weights = None

        # Flash attention if the tensors are packed
        if cu_seqlens is not None:
            attn = flash_attn_varlen_func(
                q=xq.squeeze(0),
                k=xk.squeeze(0),
                v=xv.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,
            )
        # Eager attention if attention weights are needed in the output
        elif output_attentions:
            attn_weights = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            if attention_mask is not None:
                attn_weights = attn_weights * attention_mask
            attn_weights = attn_weights.softmax(-1)
            attn = attn_weights @ xv.permute(0, 2, 1, 3)
            attn = attn.transpose(1, 2)
        # Fall back to SDPA otherwise
        else:
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attention_mask.bool() if attention_mask is not None else None,
                dropout_p=0,
            ).transpose(1, 2)

        return self.wo(attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.config.dim_head)), attn_weights


class NeoBERTPreTrainedModel(PreTrainedModel):
    config_class = NeoBERTConfig
    base_model_prefix = "model"
    _supports_cache_class = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)


class NeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # NOTE (v5 port): RoPE cos/sin are computed on the fly in forward (fp32) — no precomputed complex
        # rotary buffer. The original's non-persistent complex buffer was left uninitialized by
        # transformers-v5 meta-device loading, and its complex dtype was mangled by v5 casting.

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        self.layer_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        # Initialize
        hidden_states, attentions = [], []

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(
                1, self.config.num_attention_heads, attention_mask.size(-1), 1
            )

        # Checks to be done if inputs are packed sequences
        if cu_seqlens is not None:
            assert (
                FLASH_ATTN_AVAILABLE
            ), "Flash-attention is not available. Please ''pip install flash_attn'', or provide un-packed sequences."
            assert not output_attentions, "Output attentions is not supported when sequences are packed."
            assert max_seqlen is not None, "Missing max_seqlen. It must be provided when cu_seqlens are not None."
            assert (input_ids if input_ids is not None else inputs_embeds).shape[
                0
            ] == 1, "Cumulative sequence lengths are provided but inputs are not packed."
            assert (
                input_ids if input_ids is not None else inputs_embeds
            ).is_cuda, "Packing uses an implementation of flash-attention and is only supported on GPU."

        # Embedding
        x = self.encoder(input_ids) if input_ids is not None else inputs_embeds

        # RoPE (real cos/sin, fp32, computed on the fly — no complex buffer, v5-safe)
        cos, sin = compute_cos_sin(x.shape[1], self.config.dim_head, ROPE_THETA, x.device, position_ids)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x, attn = layer(x, attention_mask, cos, sin, output_attentions, max_seqlen, cu_seqlens)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        # Final normalization layer
        x = self.layer_norm(x)

        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
        )


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.model = NeoBERT(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        output = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.decoder(output.last_hidden_state)

        return MaskedLMOutput(
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions if output_attentions else None,
            logits=logits,
        )


class NeoBERTForSequenceClassification(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

        output = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = output.last_hidden_state

        x = hidden_states[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions if output_attentions else None,
        )
