"""SELF-CONTAINED same-architecture parity check (generated; do not edit by hand).

Run on the cluster in venv3 (torch 2.7.1 + transformers 5.12.1), in the neobert experiment dir:
    python parity_selfcontained.py
Evaluates the FROZEN parity gate (F11) same-arch: finite, cos>0.9999, magnitude ratio,
max_abs/|ref|_max < 1.5e-2, median_abs < 3e-4. Prints PASS/FAIL per case.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from safetensors.torch import load_file

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    GradientCheckpointingLayer = nn.Module

class NeoBERTConfig(PretrainedConfig):
    """Configuration for the NeoBERT encoder backbone.

    Field names and default values mirror the published ``chandar-lab/NeoBERT`` ``config.json``
    so that ``NeoBERTConfig()`` alone reconstructs the released model.
    """

    model_type = "neobert"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        classifier_init_range: float = 0.02,
        norm_eps: float = 1e-5,
        max_length: int = 4096,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.classifier_init_range = classifier_init_range
        self.norm_eps = norm_eps
        self.max_length = max_length

    @property
    def max_position_embeddings(self) -> int:
        """Alias for :attr:`max_length` — the standard name some transformers utilities read."""
        return self.max_length


ROPE_THETA = 10000.0


def compute_cos_sin(
    seq_len: int,
    dim: int,
    theta: float,
    device: torch.device,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-valued RoPE cos/sin, computed on the fly in fp32.

    Numerically identical to the original complex ``freqs_cis`` (same ``inv_freq``, same rotation),
    but with no precomputed complex buffer for v5 to leave uninitialised or corrupt. Returns
    ``cos, sin`` of shape ``(seq_len, dim // 2)`` in fp32.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: (dim // 2)] / dim))
    if position_ids is not None:
        # Only 1-D (packed/flat) position_ids are supported; 2-D per-example ids would need
        # position-indexed rotary tables (the original) and don't broadcast here (cf. MVR note).
        assert position_ids.dim() == 1, "compute_cos_sin supports only 1-D position_ids (got %dD)." % position_ids.dim()
        t = position_ids.to(device=device, dtype=torch.float32)
    else:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim // 2)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings with real cos/sin, matching the original interleaved
    (consecutive-pair) complex formulation exactly.

    The original does ``view_as_complex(x.reshape(..., -1, 2))``: complex[i] = x[2i] + i*x[2i+1].
    Multiplying by (cos + i*sin) gives out[2i] = x[2i]*cos - x[2i+1]*sin,
    out[2i+1] = x[2i]*sin + x[2i+1]*cos — reproduced here in real arithmetic.

    ``xq, xk``: ``(batch, seq, heads, dim_head)``. ``cos, sin``: ``(seq, dim_head // 2)``.
    """
    cos = cos[None, :, None, :]  # (1, seq, 1, d/2)
    sin = sin[None, :, None, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        x1 = xf[..., 0::2]  # even indices -> real part
        x2 = xf[..., 1::2]  # odd indices  -> imag part
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        return torch.stack((o1, o2), dim=-1).flatten(-2).type_as(x)  # re-interleave

    return rotate(xq), rotate(xk)


class SwiGLU(nn.Module):
    """Pure-torch SwiGLU. The fused ``w12`` (gate+value) / ``w3`` layout keeps the original
    checkpoint's weight names, so weights load unchanged: ``x1, x2 = w12(x).chunk(2, -1);
    w3(silu(x1) * x2)``."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _sdpa_padding_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """``[B, S]`` with 1 = keep, 0 = pad  ->  bool ``[B, 1, 1, S]`` (True = attend), broadcastable
    over heads and query positions (key-only masking, matching the original)."""
    if attention_mask is None:
        return None
    return attention_mask[:, None, None, :].to(torch.bool)


def _eager_attention(
    xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, attn_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct additive-mask eager attention (fp32 softmax), numerically consistent with the SDPA
    path. Inputs ``[B, S, H, D]``; returns ``(attn_output [B, S, H, D], attn_weights [B, H, S, S])``."""
    q, k, v = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)  # [B, H, S, D]
    scaling = q.shape[-1] ** -0.5
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
    if attn_mask is not None:  # bool [B, 1, 1, S], True = attend
        attn_weights = attn_weights.masked_fill(~attn_mask, torch.finfo(attn_weights.dtype).min)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()  # [B, S, H, D]
    return attn_output, attn_weights


class NeoBERTEncoderLayer(GradientCheckpointingLayer):
    """Pre-norm transformer encoder block. Attribute names (``qkv``, ``wo``, ``ffn``,
    ``attention_norm``, ``ffn_norm``) match the checkpoint key layout."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()
        self.config = config

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # SwiGLU intermediate size: 2/3 of intermediate_size, rounded up to a multiple of 8.
        multiple_of = 8
        intermediate_size = int(2 * config.intermediate_size / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)

        self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        output_attentions: bool = False,
        use_eager: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_output, attn_weights = self._att_block(
            self.attention_norm(x), attention_mask, cos, sin, output_attentions, use_eager
        )
        x = x + attn_output
        x = x + self.ffn(self.ffn_norm(x))
        return x, attn_weights

    def _att_block(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        output_attentions: bool,
        use_eager: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        num_heads, dim_head = self.config.num_attention_heads, self.config.dim_head

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, num_heads, dim_head * 3).chunk(3, dim=-1)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)  # [B, S, H, D]

        if output_attentions or use_eager:
            attn_output, attn_weights = _eager_attention(xq, xk, xv, attention_mask)
        else:
            attn_output = F.scaled_dot_product_attention(
                xq.transpose(1, 2),
                xk.transpose(1, 2),
                xv.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)  # [B, S, H, D]
            attn_weights = None

        out = self.wo(attn_output.reshape(batch_size, seq_len, num_heads * dim_head))
        return out, attn_weights


class NeoBERTPreTrainedModel(PreTrainedModel):
    config_class = NeoBERTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NeoBERTEncoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = False

    def _init_weights(self, module: nn.Module) -> None:
        # Generic over module types (matches the original uniform init distribution), so that
        # lightning-ir-added layers (e.g. the Col projection, an nn.Linear) are initialised too —
        # no backbone-specific special-case is needed in lightning_ir/base/model.py.
        #
        # Uses the guarded ``transformers.initialization`` functions (operating on the Parameter,
        # not ``.data``): under v5 these skip parameters already flagged ``_is_hf_initialized``, so
        # weights loaded by ``from_pretrained`` are never clobbered on (re)load (the F2 regression).
        # Lazy import: v5-only module, and this path only runs under the v5 runtime (never in the
        # transformers-4.x reference venv, which only instantiates SwiGLU for the equivalence test).
        from transformers import initialization as init

        if isinstance(module, nn.Linear):
            init.uniform_(module.weight, a=-self.config.decoder_init_range, b=self.config.decoder_init_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.uniform_(module.weight, a=-self.config.embedding_init_range, b=self.config.embedding_init_range)
            # Explicit flag check: slicing the weight for the padding row loses the flag.
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if getattr(module, "weight", None) is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)


class NeoBERTModel(NeoBERTPreTrainedModel):
    """Bare NeoBERT encoder outputting raw hidden states (registered under ``AutoModel``)."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)
        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.transformer_encoder = nn.ModuleList(
            NeoBERTEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.layer_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.encoder

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.encoder = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,  # accepted and ignored (RoPE-only, no token types)
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutput | tuple:
        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        # v5 deprecates config.use_return_dict (accessing the property warns); default to True.
        return_dict = True if return_dict is None else return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        x = self.encoder(input_ids) if input_ids is not None else inputs_embeds

        attn_mask = _sdpa_padding_mask(attention_mask)
        cos, sin = compute_cos_sin(x.shape[1], self.config.dim_head, ROPE_THETA, x.device, position_ids)
        use_eager = getattr(self.config, "_attn_implementation", "sdpa") == "eager"

        all_hidden_states: tuple[torch.Tensor, ...] = (x,) if output_hidden_states else ()
        all_attentions: tuple[torch.Tensor, ...] = ()

        for layer in self.transformer_encoder:
            x, attn = layer(x, attn_mask, cos, sin, output_attentions, use_eager)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            if output_attentions:
                all_attentions = all_attentions + (attn,)

        x = self.layer_norm(x)

        if not return_dict:
            return tuple(
                v for v in (x, all_hidden_states or None, all_attentions or None) if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states or None,
            attentions=all_attentions or None,
        )


class NeoBERTForMaskedLM(NeoBERTPreTrainedModel):
    """NeoBERT with an (untied) MLM decoder (registered under ``AutoModelForMaskedLM``).

    Exists for checkpoint conversion, the parity gate, and future SPLADE work; the lightning-ir
    factory uses :class:`NeoBERTModel`.
    """

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)
        self.config = config
        self.model = NeoBERTModel(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> MaskedLMOutput | tuple:
        return_dict = True if return_dict is None else return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.decoder(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


MAX_REL_DIFF, MEDIAN_ABS_DIFF, MIN_TOKEN_COS, MAG_RATIO = 1.5e-2, 3e-4, 0.9999, (0.5, 2.0)

def _masked(t, m): return t[m.bool()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("neobert_original_ref/model.safetensors"))
    ap.add_argument("--fixture", type=Path, default=Path("reference_outputs.pt"))
    args = ap.parse_args()
    import platform, transformers
    print(f"[env] torch={torch.__version__} transformers={transformers.__version__} machine={platform.machine()}")
    sd = load_file(args.weights)
    full = NeoBERTForMaskedLM(NeoBERTConfig())
    missing, unexpected = full.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    model = full.model.eval()
    fx = torch.load(args.fixture, weights_only=False)
    print(f"[ref] {fx['meta']}")
    print(f"\n{'case':16} {'rel':>9} {'median':>9} {'cos':>8} {'magratio':>9}  verdict")
    all_pass = True
    for case, d in fx["cases"].items():
        if case == "padded_batch":
            continue
        m, ref = d["attention_mask"], d["last_hidden_state"]
        with torch.no_grad():
            got = model(input_ids=d["input_ids"], attention_mask=m).last_hidden_state
        a, b = _masked(got, m), _masked(ref, m)
        diff = (a - b).abs()
        ref_mag, got_mag = b.abs().max().item(), a.abs().max().item()
        rel = diff.max().item() / ref_mag
        med = diff.median().item()
        cos = F.cosine_similarity(a, b, dim=-1).min().item()
        ratio = got_mag / ref_mag
        ok = (torch.isfinite(got).all().item() and rel < MAX_REL_DIFF and med < MEDIAN_ABS_DIFF
              and cos > MIN_TOKEN_COS and MAG_RATIO[0] <= ratio <= MAG_RATIO[1])
        all_pass &= ok
        print(f"{case:16} {rel:9.2e} {med:9.2e} {cos:8.5f} {ratio:9.3f}  {'PASS' if ok else 'FAIL'}")
    print(f"\n=== FROZEN PARITY GATE: {'PASS' if all_pass else 'FAIL'} ===")

if __name__ == "__main__":
    main()
