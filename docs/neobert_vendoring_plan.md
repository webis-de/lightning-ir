# Implementation Spec v3.1: Vendored NeoBERT backbone for lightning-ir (base: `mvrt5`)

**Audience:** a coding agent/LLM working in the `webis-de/lightning-ir` repository. Follow this spec exactly, phase by phase, in order.

**Protocol keywords:**
- **VERIFY** — run the described check and paste the raw output into your report before continuing.
- **GROUND TRUTH (GT)** — the value/name must be read from a real file (checkpoint, config.json, repo source), never assumed or recalled. Placeholders marked `<GT>` in code skeletons must be filled from ground truth.
- **PREFILLED** — a recon answer already established by reading the prior attempt branches; you must still validate it against the original source and flag any mismatch.
- **STOP** — do not guess or work around; report the exact situation (with traceback/output) and wait.
- Every phase ends with a **Definition of Done (DoD)**. Do not start the next phase until all DoD items pass.

---

## 1. Objective and repo context

Make the NeoBERT encoder (`chandar-lab/NeoBERT`; Hub-only `custom_code`, not in the transformers library) a **native, vendored backbone of lightning-ir** — usable with Col/Splade/Dpr/Coil/Mvr exactly like BERT/ModernBERT — **without** `trust_remote_code`, on **transformers v5** (the repo pins `transformers[torch]>=5.0.0` on the base branch).

**Base branch:** `origin/mvrt5` — it contains the transformers-v5 class-factory support and the MVR work; `main` does not. Create the working branch from it:

```bash
git fetch origin && git checkout -b neobert-backbone origin/mvrt5
pip install -e .[test]   # editable install in the experiment env
```

**Prior attempts (context, read-only):** branches `try_neobert` and `origin/neobertt5` contain two earlier NeoBERT integrations. Both used the **remote-code route** (resolving the backbone via `auto_map` + `get_class_from_dynamic_module`, copying dynamic-module `.py` files into saved checkpoints, defaulting `trust_remote_code=True` in tokenizer loading). That route is **superseded by this spec** — it broke under transformers v5 (RoPE buffer initialization producing NaNs in code we don't control) and required invasive special-casing in `base/class_factory.py`, `base/model.py`, and `base/tokenizer.py`. Do **not** merge or cherry-pick that machinery. The branches remain valuable as recon: their commit messages and code comments document real NeoBERT integration hazards, which are folded into this spec as PREFILLED findings (§6.1) and regression tests (§12.4).

A subsequent debugging session isolated the true NaN root cause (transformers v5 × NeoBERT's complex RoPE buffer — F6 below) and produced a **numerically reviewed patched port** of `model.py`/`rotary.py`. `docs/neobert_known_findings.md` is the authoritative post-mortem; commit it to this branch and read it before Phase 2. This spec vendors that reviewed port properly (registered classes, no `auto_map`) instead of deploying it as a local remote-code snapshot.

End state:

1. A vendored, self-contained NeoBERT implementation inside `lightning_ir/` (no `xformers`, no `flash_attn`, no remote code, no `auto_map`).
2. A locally converted checkpoint loading via `AutoModel.from_pretrained(<local_path>)` with **zero missing and zero unexpected keys**.
3. A passing **parity gate** against the original remote-code model.
4. `ColModel.from_pretrained(<local_path>)` producing a working derived model through the **unmodified** mvrt5 class factory, and a **20-step ColBERT training probe** with decreasing loss and no NaNs.
5. Regression tests covering the two failures that killed the previous attempts (§12.4).

## 2. Strategy: why vendoring, and what the factory needs

The mvrt5 class factory (`lightning_ir/base/class_factory.py`) resolves backbones purely through the transformers Auto registries:

- `LightningIRClassFactory.get_backbone_config` → `CONFIG_MAPPING[backbone_model_type]`
- `_get_model_class` → `MODEL_MAPPING[type(config)]`
- `LightningIRTokenizerClassFactory.from_pretrained` → `TOKENIZER_MAPPING[type(backbone_config)]`

Therefore: if we register `NeoBERTConfig` under `model_type="neobert"` with `AutoConfig`, `AutoModel`, **and `AutoTokenizer`**, the factory works with **zero changes** — it will build `ColNeoBERTModel` and friends via `type(name, (Mixin, NeoBERTModel), {"config_class": ..., "_backbone_forward": NeoBERTModel.forward})` exactly as it does for BERT. The tokenizer registration is mandatory: the factory looks tokenizers up by backbone config class, and its `KeyError` fallback path is what forced `trust_remote_code` in the old attempts.

Registration slots into lightning-ir's existing pattern: `lightning_ir/models/register_internal_models.py` is called from `lightning_ir/__init__.py` at import time (`_register_internal_models()`), so a sibling `_register_backbones()` runs on every `import lightning_ir` — CLI, SLURM, tests, everything. No launcher tricks needed.

Note: `lightning_ir/base/external_model_hub.py` (`CHECKPOINT_MAPPING`, `BACKBONE_MAPPING`, ...) is for loading *published external checkpoints* (naver SPLADE, colbert-ir, ...) with key remapping. It is NOT the mechanism for this task — do not touch it.

## 3. Hard constraints and guardrails

- **Do not** add/upgrade/downgrade dependencies. `xformers` and `flash_attn` live only in a throwaway reference venv (§11.1) and must never appear in `pyproject.toml`.
- **Do not** use `trust_remote_code=True` outside `scripts/neobert/dump_reference_outputs.py`.
- **Do not** modify `base/class_factory.py`, `base/model.py`, or `base/tokenizer.py`. The design goal is that registration alone suffices. If a phase seems to require touching them, STOP with the traceback — that signals a bug in the vendored model, not a missing factory feature. (The single allowed edit outside new files: the one-line `ADD_MARKER_TOKEN_MAPPING` entry, §12.2, plus the `__init__.py` registration hook, §12.1.)
- **Do not** change the numerics of the model relative to the original (computation order, norm placement, eps, dtype casts).
- **Do not** loosen tolerances, skip a VERIFY, or self-certify a checkbox without output.
- Licensing: original code/weights are MIT. Every vendored file gets a header: origin (`chandar-lab/NeoBERT`, snapshot revision), MIT notice, one-line list of modifications.
- Style: type hints, short docstrings, no dead code; match the conventions of `lightning_ir/models/`.

## 4. Phase −1 — Environment verification

```bash
python -c "import sys; print(sys.version)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"   # must be >= 5.0.0
python -c "import lightning_ir, os; print(lightning_ir.__version__, os.path.dirname(lightning_ir.__file__))"
git branch --show-current   # must be neobert-backbone
git merge-base --is-ancestor origin/mvrt5 HEAD && echo "based on mvrt5: OK"
```

- VERIFY the lightning-ir source path points into this working tree (editable install); if it points into site-packages, `pip install -e .` and re-check.
- If transformers is not v5.x: STOP (wrong environment for this branch).

**DoD:** version block recorded; branch based on mvrt5 confirmed; editable install verified.

## 5. Deliverables (file layout)

```
<repo_root>/
  third_party/neobert_original/                  # pristine Hub snapshot; read-only; add to .gitignore (weights)
  docs/neobert_known_findings.md                 # debugging post-mortem (provided by Wilhelm); authoritative for F6-F10
  lightning_ir/models/backbones/__init__.py      # new subpackage
  lightning_ir/models/backbones/neobert/
    __init__.py
    configuration_neobert.py
    modeling_neobert.py                          # includes rotary + swiglu (or sibling rotary.py)
  lightning_ir/models/register_backbones.py      # _register_backbones(), mirrors register_internal_models.py
  scripts/neobert/
    convert_checkpoint.py
    dump_reference_outputs.py                    # ONLY run in the pinned reference venv
    reference_state_dict_keys.txt                # generated in Phase 0
  tests/test_backbones/
    __init__.py
    fixtures/reference_outputs.pt                # generated in §11.2; gitignore if large, document regeneration
    test_swiglu_equivalence.py                   # importorskip("xformers"): runs only in reference venv
    test_neobert_unit.py                         # state-dict load, roundtrip, init-weights behavior (CPU, CI-safe)
    test_neobert_parity.py                       # skipped unless fixture + converted checkpoint exist
    test_neobert_lightning_ir.py                 # factory/Col integration + regression tests (§12.4)
  configs/examples/colbert_probe_neobert.yaml
  checkpoints/neobert-vendored/                  # output of convert_checkpoint.py; gitignore
```

Before creating anything, check the layout against the repo's conventions (tests live in `tests/test_models/*.py` with `@pytest.mark.model` and `--run-models` gating in `tests/conftest.py` — mirror those gating conventions so CI stays green without the fixture, checkpoint, or GPU). Record any deviation (spec path → actual path).

## 6. Phase 0 — Recon (read-only)

### 6.1 PREFILLED findings from the prior attempts (validate, don't re-derive blindly)

From `origin/neobertt5` (diff vs `origin/mvrt5`), its commit history, and the debugging post-mortem `docs/neobert_known_findings.md`:

- **F1 — input embeddings:** NeoBERT stores its token embedding under an attribute the standard API doesn't expect (the old attempt's comment: "NeoBERT stores its token embedding as `self.encoder` and does not implement the standard get/set_input_embeddings, so resize_token_embeddings cannot find the table"). Consequence: the vendored `NeoBERTModel` MUST implement `get_input_embeddings`/`set_input_embeddings` correctly (§8.6) — ColBERT's marker tokens require `resize_token_embeddings` to work.
- **F2 — weight init clobbering:** NeoBERT's original `_init_weights` initializes in place and bypasses transformers' `_is_hf_initialized` gate; on the reload path transformers then re-initialized already-loaded weights (the "re init weights" fix in the old branch). Consequence: the vendored port must implement a standard per-module `_init_weights` (§8.7) and §12.4 adds a reload-no-clobber regression test.
- **F3 — uninitialized heads:** ModernBERT-style `_init_weights` only initializes the backbone's own module types, leaving lightning-ir-added layers (e.g. the Col projection) as uninitialized memory → "huge values and inf scores" (comment in the old branch; `base/model.py` on mvrt5 still special-cases `modernbert` for this). Consequence: the vendored `_init_weights` must be **generic over module types** (§8.7) so that NO `neobert` special-case in `base/model.py` is needed. §12.4 tests this.
- **F4 — marker tokens:** ColBERT-style query/doc markers need an entry in `ADD_MARKER_TOKEN_MAPPING` (`lightning_ir/bi_encoder/bi_encoder_tokenizer.py`); the old branch added `"neobert": {"pattern": "[CLS] {TOKEN} $0 [SEP]", "special_tokens": ["[CLS]", "[SEP]"]}`. This one line is the only thing to carry over verbatim (§12.2). It also implies NeoBERT uses a BERT-style tokenizer with `[CLS]`/`[SEP]` — validate in §6.4.
- **F5 — tokenizer resolution:** the factory's `TOKENIZER_MAPPING[type(backbone_config)]` lookup raised `KeyError` for the unregistered NeoBERT config, which is what dragged `trust_remote_code` into `base/tokenizer.py` in the old attempt. Consequence: `AutoTokenizer.register(NeoBERTConfig, ...)` in §12.1 — with it, the `KeyError` path is never taken.

From the debugging post-mortem — these are **confirmed root-cause facts**, not hypotheses:

- **F6 — the v5 NaN root cause (two independent bugs, both around the complex RoPE buffer):** the original code precomputes a `complex64` `freqs_cis` buffer in `__init__`, registered `persistent=False`. Under transformers v5, (a) meta-device loading never materializes non-persistent buffers (not in the checkpoint) → per-process uninitialized garbage/NaN, and (b) v5's dtype-cast machinery corrupts complex buffers even when finite. A/B confirmed: same model, same input — transformers 4.57.6 finite, 5.12.1 NaN. **Design mandate: the vendored port has NO RoPE buffer of any kind** — cos/sin are computed on the fly in fp32 inside `forward`. `persistent=False` alone is NOT a fix.
- **F7 — rotation convention:** NeoBERT's RoPE is the **interleaved consecutive-pair** rotation (`x[..., 0::2]`, `x[..., 1::2]`), equivalent to `view_as_complex` — the reviewed replacement matches the complex reference to machine epsilon (2.2e-16), while a `rotate_half`-style split diverges by ~3.8 yet looks plausible. Use the replacement verbatim (§8.2).
- **F8 — SwiGLU layout confirmed:** packed `w12` (`2*hidden`) + `w3`; forward `x1, x2 = w12(x).chunk(2, -1); w3(F.silu(x1) * x2)`. Loads the hub checkpoint unchanged.
- **F9 — parity expectations:** reference = hub remote code on transformers 4.57.6. Expected `max_abs_diff` ≈ 1e-4–1e-3 (torch-vs-xformers SwiGLU kernels + SDPA backend rounding). The historical ~8000-scale residual activations were **buffer corruption, not a model property** — the gate therefore also checks activation magnitude (§11.3).
- **F10 — a reviewed port already exists:** patched `model.py`/`rotary.py` from the debugging session (the local `neobert-v5` snapshot). Phase 2 starts from those files; the hub snapshot stays the GT reference for recon. If the files are not in the repo or provided, STOP and ask for them before writing modeling code.

### 6.2 Download the snapshot

```python
from huggingface_hub import snapshot_download, HfApi
p = snapshot_download("chandar-lab/NeoBERT", local_dir="third_party/neobert_original")
print(p, HfApi().model_info("chandar-lab/NeoBERT").sha)   # record the revision
```

### 6.3 Read the original code fully

Read `model.py` and `rotary.py` end to end. Produce a table: class name, constructor args, submodule attribute names, external imports. Then answer, each with a line reference (GT):

1. Is the QKV projection fused (one `Linear` for q,k,v) or three separate `Linear`s? Attribute name?
2. What FFN module is used, imported from where, with what constructor call?
3. Which norm (RMSNorm/LayerNorm), pre- or post-attention/FFN, which eps, final norm after the last layer?
4. Token type embeddings? Absolute position embeddings? (Expected: RoPE only.) What is the token-embedding attribute name — validate PREFILLED F1.
5. Does `forward` accept/use `token_type_ids`? (The BERT-style tokenizer emits them regardless — see §8.1.)
6. How is the padding mask consumed (additive float / bool / xformers BlockDiagonalMask for packing), and which code path is the unpadded/eager one to reproduce?
7. Attention/hidden dropout, with which config fields?
8. RoPE: which function, applied to q/k where, rotation convention (rotate-half vs interleaved), cos/sin origin and dtype, tensor layout at application time?
9. Is the MLM decoder tied to the input embedding (code AND checkpoint: separate decoder weights present)?
10. What head classes exist beyond the base model, and which matches the checkpoint keys? What is `base_model_prefix` / the top-level attribute nesting?

Additionally: read the original `_init_weights` and describe exactly how it differs from the transformers per-module convention — validate PREFILLED F2/F3.

### 6.4 State dict keys, config, tokenizer (GT anchors)

```python
from safetensors.torch import load_file
sd = load_file("third_party/neobert_original/model.safetensors")
with open("scripts/neobert/reference_state_dict_keys.txt", "w") as f:
    for k in sorted(sd):
        f.write(f"{k}\t{tuple(sd[k].shape)}\t{sd[k].dtype}\n")
```

VERIFY: paste first 30 lines + total count + weight dtype. Paste the full original `config.json`. Then:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("third_party/neobert_original")   # NO trust_remote_code
print(type(tok).__name__, tok.padding_side, tok("hello world"))
```

- If this fails without remote code: STOP (the tokenizer plan depends on a standard class).
- Record: exact tokenizer class (expected BERT-family per F4), whether `token_type_ids` are emitted, pad/cls/sep/mask ids, `padding_side` (expected `right`).

**DoD:** all questions answered with line references; PREFILLED F1–F4 validated or corrected; keys file written; config pasted; tokenizer class recorded.

## 7. Phase 1 — `configuration_neobert.py`

```python
from transformers import PretrainedConfig


class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    def __init__(
        self,
        vocab_size: int = <GT>,
        hidden_size: int = <GT>,
        num_hidden_layers: int = <GT>,
        num_attention_heads: int = <GT>,
        intermediate_size: int = <GT>,
        # keep ORIGINAL field names (e.g. max_length) with original values as defaults: <GT>
        norm_eps: float = <GT>,
        rope_theta: float = <GT>,
        pad_token_id: int = <GT>,
        # every other field in the original config.json: <GT>
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # assign all fields; assert hidden_size % num_attention_heads == 0
```

Rules:

- Constructor args = exactly the fields of the original `config.json`, with the original values as defaults, so `NeoBERTConfig()` reproduces the published architecture. Unknown kwargs route to `super().__init__` (mandatory: the factory builds `type(..., (BiEncoderConfigMixin, NeoBERTConfig), {...})` with the mixin's `__init__`, which chains into `NeoBERTConfig.__init__(**kwargs)` cooperatively).
- Add alias properties for standard names the rest of the stack reads (`max_position_embeddings`, etc.) if the original names differ. GT which attributes are needed: grep `lightning_ir/bi_encoder/` and `lightning_ir/models/` for `config.` accesses on backbone attributes (`hidden_size` is certainly read for projections; `vocab_size` by resize).

**DoD:** `NeoBERTConfig()` constructs; `.to_dict()` has `model_type == "neobert"` and all original fields; required attribute accesses resolve.

## 8. Phase 2 — `modeling_neobert.py`

**Starting point (F10):** adapt the reviewed patched `model.py`/`rotary.py` from the debugging session into the vendored classes below — do not re-port from the hub snapshot. The adaptation work is: wrap them as proper `PreTrainedModel` subclasses with `NeoBERTConfig`, add the Phase 4 v5 compliance, and keep every numerical decision of the patched files intact.

Mirror the original module hierarchy (attribute names, nesting) so state dict keys match with minimal or no remapping. When "nicer code" conflicts with matching keys, match the keys. Constructors take `(self, config)` only and call `super().__init__(config)` cooperatively (the factory builds the derived model by multiple inheritance `(Mixin, NeoBERTModel)`).

### 8.1 Embeddings

Token embedding only (per §6.3 Q4; follow GT if different). `forward` must **accept** `token_type_ids=None` and ignore it if unused — the BERT-family tokenizer emits it and the derived model's `_backbone_forward` receives whatever the lightning-ir tokenizer produced; a `TypeError` here is exactly the class of bug the old branch's "quick fix for backbone tokenizer" papered over.

### 8.2 RoPE (root-cause area — follow F6/F7 exactly)

Do **not** port the original `rotary.py` — it is the buggy complex-buffer implementation. Use the reviewed replacement:

```python
def compute_cos_sin(seq_len, dim, theta, device, position_ids=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: dim // 2] / dim))
    if position_ids is not None:
        assert position_ids.dim() == 1     # only 1-D supported for now (see Phase 8 MVR note)
        t = position_ids.to(device=device, dtype=torch.float32)
    else:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)       # (seq_len, dim/2)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(xq, xk, cos, sin):
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    def rotate(x):
        xf = x.float()
        x1, x2 = xf[..., 0::2], xf[..., 1::2]   # consecutive-pair: matches view_as_complex (F7)
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).type_as(x)

    return rotate(xq), rotate(xk)
```

Rules:

- **No RoPE buffer of any kind** — not even `persistent=False` (F6): compute `cos, sin` in `NeoBERTModel.forward` after the embedding (fp32) and thread them into every layer, replacing the original `freqs_cis` argument. Nothing exists for v5's meta-device loading to leave uninitialized, and no complex dtype exists for the cast machinery to mangle.
- The interleaved consecutive-pair rotation is mandatory (F7); do not "simplify" to rotate-half.
- `theta` and head dim from config (GT; the original names the head dim `dim_head`, theta 10000.0 per the post-mortem — verify against `config.json`).
- Note the rotary application layout implied by the broadcast above is `[B, S, H, D]` (heads on dim 2) — keep the patched files' layout; if it differs from §8.3's `[B, H, S, D]` skeleton, the skeleton yields to the patched files.
- `position_ids`: `None` for padded batches → `arange`. Right padding gives pad positions valid ids — fine, they're masked out; do not invent left-padding handling.

### 8.3 Attention (SDPA replaces xformers/flash paths)

```python
def _sdpa_padding_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """[B, S] with 1 = keep, 0 = pad  ->  bool [B, 1, 1, S], True = attend."""
    if attention_mask is None:
        return None
    return attention_mask[:, None, None, :].to(torch.bool)
```

Forward (adapt names to §6.3 Q1):

```python
B, S, _ = hidden_states.shape
qkv = self.qkv(hidden_states)                    # if fused; else three projections
q, k, v = qkv.chunk(3, dim=-1)                   # VERIFY chunk order against the original slicing
q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)   # or original layout, per GT
k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
q, k = apply_rotary(q, k, cos, sin)              # exactly the vendored rotary fn / layout
attn = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=_sdpa_padding_mask(attention_mask),
    dropout_p=self.attention_dropout if self.training else 0.0,   # <GT>; 0.0 if none
    is_causal=False,
)
attn = attn.transpose(1, 2).reshape(B, S, -1)
out = self.o_proj(attn)                          # attribute name <GT>
```

Bidirectional, no sliding window, no packing. Remove all `flash_attn` imports and packed-sequence paths.

### 8.4 SwiGLU FFN (replaces `xformers.ops.SwiGLU`)

Decide packed vs unpacked from the keys file (`...ffn.w12.weight` + `...ffn.w3.weight` → packed; `w1/w2/w3` → unpacked). Implement the matching variant with **identical parameter names**:

```python
class SwiGLU(nn.Module):
    """Pure-PyTorch replacement for xformers.ops.SwiGLU (packed variant).
    Parameter names (w12, w3) match xformers so the checkpoint loads without remapping."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = <GT>):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)   # gate order CONFIRMED (F8): silu gates the first chunk
```

`tests/test_backbones/test_swiglu_equivalence.py` (reference venv only, `pytest.importorskip("xformers")`): instantiate `xformers.ops.SwiGLU(<GT dims>)` and the replacement, `load_state_dict` from one to the other **as-is** (name mismatch = fix names, don't remap), print both state dict key lists, `torch.testing.assert_close(atol=1e-6, rtol=1e-5)` on random fp32 input **on CPU** (fused GPU kernels add larger diffs; CPU isolates the math). This validates F8 rather than discovering it.

### 8.5 Layer and model assembly

- `NeoBERTLayer` (name per original): pre-norm blocks exactly as the original (`x = x + attn(norm1(x)); x = x + ffn(norm2(x))` or whatever GT says), original norm class/eps, final norm iff its weight is in the keys file. RMSNorm: `torch.nn.RMSNorm` if the installed torch has it, else a local implementation with the original eps.
- `NeoBERTModel.forward(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None) -> BaseModelOutput`. `input_ids` xor `inputs_embeds`; `hidden_states` tuple (embedding output + after each layer) when requested; gradient checkpointing per §10.
- `NeoBERTForMaskedLM` (or the exact head class the checkpoint contains, §6.3 Q10): original head structure per the keys file, accepts `labels` (`CrossEntropyLoss(ignore_index=-100)`), returns `MaskedLMOutput`. This class exists for conversion/parity and future SPLADE work; lightning-ir's factory uses `NeoBERTModel`.

### 8.6 `get_input_embeddings` / `set_input_embeddings` (fixes F1)

Implement both on `NeoBERTModel` (and delegate from the MLM class), returning/setting the actual token-embedding module whatever the original calls it. Then `resize_token_embeddings` — which ColBERT's marker tokens exercise — works through the standard `PreTrainedModel` path, and the old branch's `base/model.py` override becomes unnecessary. If §6.3 Q9 says weights are tied, also implement `get_output_embeddings`/`set_output_embeddings` on the MLM class and set `config.tie_word_embeddings` accordingly.

### 8.7 `_init_weights` (fixes F2 + F3)

On `NeoBERTPreTrainedModel`, implement the standard transformers per-module convention, **generic over module types**:

```python
def _init_weights(self, module: nn.Module) -> None:
    std = <GT: the original init std, else 0.02>
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        if getattr(module, "bias", None) is not None:
            module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

Rationale (report this): (a) generic branches mean layers added by lightning-ir on top of the backbone (Col projection etc.) get initialized too — no `"neobert"` special-case needed in `base/model.py`, unlike ModernBERT; (b) the per-module convention respects transformers' `_is_hf_initialized` gating, so loaded weights are never clobbered on reload — the F2 failure. Match the original init distribution where the original specifies one (GT), because from-scratch MVR/Col heads sample from it. Call `self.post_init()` at the end of each top-level model `__init__`.

**DoD:** module imports cleanly in the main env (VERIFY `python -c "import lightning_ir.models.backbones.neobert.modeling_neobert"` — no xformers/flash at import); random-init CPU forward on a padded batch works; `output_hidden_states=True` returns `num_hidden_layers + 1` tensors; `resize_token_embeddings(vocab+2)` works on a random-init instance.

## 9. Phase 3 — Checkpoint conversion

`scripts/neobert/convert_checkpoint.py`:

1. Load `third_party/neobert_original/model.safetensors`; instantiate `NeoBERTForMaskedLM(NeoBERTConfig())`.
2. Print the symmetric difference of key sets (`only in original` / `only in vendored`, with shapes). Build `ORIG_TO_VENDORED: dict[str, str]` **only** from those lists (ideally identity/empty).
3. `missing, unexpected = model.load_state_dict(converted, strict=False)`; `assert not missing and not unexpected`. Also compare shapes for every key, not just names — a name match with a shape mismatch means a wrong architecture constant: STOP.
4. `model.save_pretrained("checkpoints/neobert-vendored", safe_serialization=True)`; copy all tokenizer files from the snapshot (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` — whichever exist); do NOT copy `model.py`/`rotary.py`.
5. VERIFY the written `config.json`: `"model_type": "neobert"`, **no `auto_map` key** (delete post-hoc if it sneaks in via kwargs), architecture fields match the original values; preserve the original weight dtype.
6. VERIFY `AutoTokenizer.from_pretrained("checkpoints/neobert-vendored")` resolves the standard tokenizer class from §6.4 without remote code (if `tokenizer_config.json` names a custom class: STOP).

`tests/test_backbones/test_neobert_unit.py` (CPU, skipped if `checkpoints/neobert-vendored` absent): (a) `from_pretrained(..., output_loading_info=True)` → no missing/unexpected keys; (b) roundtrip `save_pretrained(tmp)` → `from_pretrained(tmp)` → all tensors `torch.equal`.

**DoD:** symmetric-difference output pasted; strict-equivalent load passes; both unit tests pass; config/tokenizer VERIFYs pasted.

## 10. Phase 4 — transformers v5 compliance

Resolution rule for every ambiguity: **mirror ModernBERT in the installed transformers v5 source** (closest in-tree analogue: RoPE encoder, recent API). Locate it via the path from Phase −1.

- `NeoBERTPreTrainedModel`: `config_class = NeoBERTConfig`; `base_model_prefix = <GT from keys/nesting, §6.3 Q10>`; `supports_gradient_checkpointing = True`; `_no_split_modules = ["<layer class>"]`; `_supports_sdpa = True`; `_supports_flash_attn_2 = False`.
- Attention implementation selection: make `"sdpa"` supported and default under v5's `config._attn_implementation` handling / attention-registry API — copy ModernBERT's pattern exactly.
- Gradient checkpointing via the standard flag + `self._gradient_checkpointing_func(layer.__call__, ...)` in the encoder loop when `self.gradient_checkpointing and self.training`.
- Outputs as proper dataclasses (`BaseModelOutput`/`MaskedLMOutput`); `output_attentions=True` either gets an eager fallback or raises `NotImplementedError` with a clear message — never silently `None`.
- Replace any `config.use_return_dict` access with `getattr(config, "use_return_dict", True)` — v5 deprecates the attribute (post-mortem).
- v5 tied-weights bookkeeping: if the class-level tied/ignore-key attributes ModernBERT sets exist in the installed version, set them consistently (the mvrt5 test suite already shims third-party models for exactly this — see `tests/test_models/test_col.py` — the vendored model must not need shims).

**DoD:** `from_pretrained` works with default and explicit `attn_implementation="sdpa"`; gradient-checkpointing smoke test passes (enable → forward+backward → finite grads); MLM forward with labels returns a finite scalar.

## 11. Phase 5 — Reference outputs and parity gate

### 11.1 Reference venv (the ONLY place remote code and xformers run)

```bash
python -m venv /tmp/neobert_ref && source /tmp/neobert_ref/bin/activate
export PYTHONNOUSERSITE=1   # mandatory on the cluster — ~/.local shadowing broke earlier envs
pip install "transformers==4.57.6" torch xformers safetensors pytest   # 4.57.6 = the proven finite reference
```

Record the resolved torch/xformers versions. If pip cannot resolve these on this machine, STOP with the exact error. Run `test_swiglu_equivalence.py` here and record the result.

### 11.2 `scripts/neobert/dump_reference_outputs.py` (reference venv only)

Load tokenizer + `AutoModel.from_pretrained("third_party/neobert_original", trust_remote_code=True, torch_dtype=torch.float32).eval()` on CPU. Hardcoded input battery:

- **short** (~10 tokens); **medium** (~200 tokens); **long** (~1500 tokens, exceeds 512 to exercise RoPE range); **padded_batch** (4 texts of clearly different lengths, `padding=True`, VERIFY right padding).

For each case, `torch.no_grad()`, `output_hidden_states=True` (if the remote model rejects the flag, store `last_hidden_state` only and note the degraded per-layer localization), save `input_ids`, `attention_mask`, all hidden states, `last_hidden_state`, plus a `meta` block (versions, snapshot revision, dtype) to `tests/test_backbones/fixtures/reference_outputs.pt`.

### 11.3 `tests/test_backbones/test_neobert_parity.py` (main env; skipped unless fixture + checkpoint exist)

```python
def _masked(t, mask): return t[mask.bool()]
def max_abs_diff(a, b, mask): return (_masked(a, mask) - _masked(b, mask)).abs().max().item()
def min_token_cos(a, b, mask):
    a, b = _masked(a, mask), _masked(b, mask)
    return F.cosine_similarity(a, b, dim=-1).min().item()
```

Vendored model fp32 `.eval()`, same inputs; per case, on non-pad positions of `last_hidden_state`:

- everything finite: `torch.isfinite(out.last_hidden_state).all()` — the historical failure mode (F6),
- `max_abs_diff < 2e-3`, with the expected band 1e-4–1e-3 (F9: SwiGLU-kernel + SDPA-backend rounding). Above 2e-3: debug, don't relax,
- `min_token_cos > 0.9999`,
- magnitude sanity (F9): the port's `last_hidden_state.abs().max()` within [0.5×, 2×] of the reference's, and nowhere near O(8000).

If per-layer states exist, also compare layer 0 and layer `n//2`; on any failure print the per-layer `layer → max_abs_diff` table. Gradient sanity: medium case, dummy loss `last_hidden_state.pow(2).mean()`, backward, every grad finite and not all-zero.

**Gate rule:** tolerances fixed. On failure follow Appendix A; still failing → STOP with the table.

**DoD:** 4/4 cases pass; gradient sanity passes; numbers pasted.

## 12. Phase 6 — Registration and lightning-ir integration

### 12.1 `lightning_ir/models/register_backbones.py`

Mirror `register_internal_models.py`:

```python
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .backbones.neobert import NeoBERTConfig, NeoBERTForMaskedLM, NeoBERTModel


def _register_backbones():
    AutoConfig.register(NeoBERTConfig.model_type, NeoBERTConfig, exist_ok=True)
    AutoModel.register(NeoBERTConfig, NeoBERTModel, exist_ok=True)
    AutoModelForMaskedLM.register(NeoBERTConfig, NeoBERTForMaskedLM, exist_ok=True)
    AutoTokenizer.register(
        NeoBERTConfig,
        slow_tokenizer_class=<GT from §6.4, e.g. BertTokenizer>,
        fast_tokenizer_class=<GT, e.g. BertTokenizerFast>,
    )
```

(GT the exact `register` signatures in the installed v5 — if `exist_ok` is unsupported anywhere, wrap in the narrow already-registered `ValueError` catch. The factory does `TOKENIZER_MAPPING[type(backbone_config)]`, so the AutoTokenizer registration is what keeps `base/tokenizer.py` untouched — F5.)

Call `_register_backbones()` from `lightning_ir/__init__.py`, directly next to the existing `_register_internal_models()` / `_register_external_models()` calls. That is the entire wiring.

### 12.2 Marker tokens (F4)

Add exactly one line to `ADD_MARKER_TOKEN_MAPPING` in `lightning_ir/bi_encoder/bi_encoder_tokenizer.py`:

```python
"neobert": {"pattern": "[CLS] {TOKEN} $0 [SEP]", "special_tokens": ["[CLS]", "[SEP]"]},
```

(Validate the special tokens against §6.4; if NeoBERT's tokenizer uses different specials, follow GT.)

### 12.3 Resolution smoke test

VERIFY in the main env:

```python
import lightning_ir  # triggers registration
from transformers import AutoConfig, AutoModel, AutoTokenizer
cfg = AutoConfig.from_pretrained("checkpoints/neobert-vendored")
m = AutoModel.from_pretrained("checkpoints/neobert-vendored")
t = AutoTokenizer.from_pretrained("checkpoints/neobert-vendored")
print(type(cfg).__name__, type(m).__name__, type(t).__name__)
# expected: NeoBERTConfig NeoBERTModel <the GT fast tokenizer class>

from lightning_ir.models import ColConfig
from lightning_ir import BiEncoderModule
module = BiEncoderModule(model_name_or_path="checkpoints/neobert-vendored", config=ColConfig())
print(type(module.model).__name__)   # expected: ColNeoBERTModel (factory-derived)
```

### 12.4 Regression tests (`tests/test_backbones/test_neobert_lightning_ir.py`)

These encode the exact failures of the previous attempts; all CPU, skipped if the converted checkpoint is absent:

1. **Factory derivation:** building a Col model over the vendored backbone (as in §12.3) succeeds; a forward pass on a toy tokenized batch returns finite embeddings.
2. **Head initialization (F3):** the derived model's lightning-ir-added parameters (e.g. `projection`) are initialized sanely — finite, non-zero variance, `abs().max()` below a loose bound (e.g. < 10). This is the "inf scores" regression.
3. **Reload no-clobber (F2):** `save_pretrained(tmp)` on the derived model, reload via the lightning-ir path, assert a sample of backbone tensors is `torch.equal` between saved and reloaded. This is the "re init weights" regression.
4. **Marker tokens / resize (F1+F4):** construct the Col tokenizer with marker tokens for the neobert backbone; `resize_token_embeddings` grows the vocab; original embedding rows unchanged (`torch.equal` on the first `vocab_size` rows). (Per the post-mortem the default Col setup adds no new tokens — this test is the API guarantee that MVR's `[VIE*]` viewer tokens depend on.)
5. **No factory edits:** `git diff --name-only origin/mvrt5..HEAD -- lightning_ir/base/` shows nothing (or only whitespace-free zero diff). Paste the output.

**DoD:** §12.3 output pasted; all five tests pass; `grep -rn "xformers\|flash_attn\|trust_remote_code" lightning_ir/ configs/` output pasted (no hits; the only permitted `trust_remote_code` lives in `scripts/neobert/dump_reference_outputs.py`).

## 13. Phase 7 — 20-step ColBERT probe

1. Base the probe YAML on the known-good ColBERT config used for the existing BERT-baseline runs (it may live outside this repo in the experiment setup — if not found, STOP and ask; `configs/examples/mvr_conf.yaml` shows the schema style: jsonargparse `trainer`/`data`/`model: {class_path: lightning_ir.BiEncoderModule, init_args: ...}`/`optimizer`/`lr_scheduler`). Save as `configs/examples/colbert_probe_neobert.yaml`. **Do not write it from scratch.**
2. Change ONLY: backbone `model_name_or_path` → absolute path to `checkpoints/neobert-vendored`; `trainer.max_steps: 20`; `log_every_n_steps: 1`; disable validation; scratch output dirs; batch size that fits one GPU (start at the baseline's; halve on OOM; record). Keep the baseline precision setting.
3. Launch the same way existing runs are launched (`lightning-ir fit --config ...`) — plain `import lightning_ir` performs registration, so no special launcher. Run interactively on a GPU node first (srun/tmux on the cluster); capture full stdout/stderr.
4. VERIFY and report: loss at step 1 vs step 20 (overall decrease; per-step wiggle fine); all losses/grad-norms finite; checkpoint written; **reload check** — load the written checkpoint the way the repo's evaluation path does and run one scoring forward (this exercises the saved derived config with `model_type` = the Col type and `backbone_model_type: neobert`, i.e. the factory's saved-checkpoint path).
5. Any traceback referencing `model_type`, the class factory, or attention implementation: STOP with traceback + source location (per §3, the presumption is a vendored-model bug, not a factory gap).

Cluster/ops requirements (from the post-mortem — non-negotiable):

- Run inside a venv with `export PYTHONNOUSERSITE=1`; never `pip install --user`. The persistent `~/.local` previously shadowed the container env (mangled transformers install, NCCL-mismatched torch).
- Install lightning-ir into the pinned env with `pip install -e . --no-deps`; never install unpinned `xformers`/`torchvision` (that produced the broken torch 2.11 / xformers 0.0.35 stack). After the vendored port, xformers is not needed anywhere in the main env.
- GPU sizing: `--gres=gpu:hopper:1` (80 GB) fits Col at `train_batch_size: 16`; Ampere (40 GB) OOMs → batch 8 with doubled `accumulate_grad_batches` so the effective batch stays 64 for comparability.
- Known-working main env from the investigation: torch 2.7.1+cu128, transformers 5.12.1.

**DoD:** log quoted (step-1/step-20 losses), reload verified, YAML `diff` vs baseline pasted.

## 14. Phase 8 — follow-up integration points (AFTER the probe gate; separate commits; optional in this task)

- **SPLADE:** `lightning_ir/modeling_utils/lm_head.py` needs three `"neobert"` entries: `MODEL_TYPE_TO_LM_HEAD` (a `partial(LMHead, hidden_dim_key=..., activation_key=..., ...)` matching NeoBERT's MLM-head structure — if NeoBERT's head differs structurally from dense→act→norm→decoder, STOP and report before forcing it), `MODEL_TYPE_TO_STATE_DICT_KEY_MAPPING` (original MLM-head keys → `<base_model_prefix>.projection.*`, GT from the keys file), and `MODEL_TYPE_TO_INPUT_EMBEDDINGS_KEY` (path of the token-embedding weight relative to the backbone, GT — cf. F1). The mvrt5 SPLADE decoder-tying fix consumes these tables.
- **MVR:** `MvrModel.__init__` special-cases ModernBERT only to disable local attention — NeoBERT has global attention, so likely no-op; but the RoPE viewer-token position handling must be checked (RoPE at position 0 was the ModernBERT view-collapse cause; `MvrViewCollapseCallback` exists for monitoring). Run a 20-step MVR probe before any MVR × NeoBERT experiments and watch that callback. Note: the replacement rotary asserts 1-D `position_ids`; if MVR viewer tokens need per-sequence 2-D position ids, extend `compute_cos_sin` then, with a parity-preserving test — and remember MVR is what actually exercises the resize path (§12.4 test 4).
- **Upstreaming (later):** transformers issue #37015 ("Add NeoBERT") is open with no PR; the vendored v5 port plus the 4.57.6-vs-5.12.1 A/B repro is exactly what upstream is missing. Separate task after the paper.
- Neither item blocks the ColBERT gate; implement each with its own smoke test when needed.

## 15. Final acceptance checklist (paste evidence per line)

- [ ] Phase −1: versions; branch off `origin/mvrt5`; editable install.
- [ ] Recon: table + 10 GT answers with line refs; F1–F5 validated/corrected; keys file; config.json; tokenizer class.
- [ ] SwiGLU equivalence passed in reference venv (atol 1e-6); gate order confirmed.
- [ ] Conversion: symmetric difference pasted; zero missing/unexpected; saved config `model_type: neobert`, no `auto_map`; tokenizer resolves without remote code.
- [ ] Unit tests (load, roundtrip) pass.
- [ ] Parity: 4/4 (finite; `max_abs_diff < 2e-3`, expected 1e-4–1e-3; `min_token_cos > 0.9999`; sane magnitude; non-pad); gradient sanity.
- [ ] Registration via `_register_backbones()` wired in `lightning_ir/__init__.py`; §12.3 resolution output pasted.
- [ ] Regression tests 1–5 (§12.4) pass; `lightning_ir/base/` untouched vs `origin/mvrt5`; grep for xformers/flash_attn/trust_remote_code clean.
- [ ] Probe: loss step 1 > step 20, all finite, checkpoint reloads through the factory's saved-checkpoint path; YAML diff pasted.
- [ ] Every vendored file has the origin/MIT/modifications header.

## Appendix A — Debugging decision tree

| # | Symptom | Most likely cause | First check |
|---|---------|-------------------|-------------|
| P1 | strict load: missing/unexpected keys | wrong attribute names / persistent RoPE buffers / wrong head class | keys file vs `model.state_dict()`; `persistent=False` on rotary buffers; §6.3 Q10 |
| P2 | only `padded_batch` fails parity | mask conversion (bool vs additive, broadcast dim) | `_sdpa_padding_mask` shape `[B,1,1,S]`; rerun each text with batch 1 — singles passing ⇒ mask |
| P3 | only `long` fails | RoPE cache length / position dtype / layout | cache regenerates ≥ seq_len; fp32 inv_freq; compare vendored vs original rotary fn on a probe tensor |
| P4 | all cases fail by small ~constant (1e-3–1e-2) | norm eps, norm type, SwiGLU gate order, dtype casts | config eps vs code; rerun SwiGLU micro-test; force fp32 end-to-end |
| P5 | all cases fail badly, layer 0 already off | key mapping loaded wrong modules / qkv chunk order | layer-0 diff ⇒ embeddings/first norm; check qkv order vs original slicing |
| P6 | probe NaN in bf16, fp32 parity passed | norm/softmax numerics in low precision | mirror ModernBERT's fp32-norm handling; rerun probe in fp32 to localize |
| P7 | probe loss flat | frozen backbone, LR 0, or wrong class silently resolved | print `type(module.model)` at start; `requires_grad` on backbone params; re-diff YAML |
| P8 | inf/huge scores at probe start | lightning-ir head uninitialized (F3) | §12.4 test 2; `_init_weights` generic branches actually hit `projection` |
| P9 | eval after reload much worse than training | weights clobbered on reload (F2) | §12.4 test 3; `_init_weights` respects per-module convention |
| P10 | `KeyError` in tokenizer factory / interactive remote-code prompt in SLURM | AutoTokenizer registration missing or not executed (F5) | `import lightning_ir` before loading; `TOKENIZER_MAPPING` contains `NeoBERTConfig` |
| P11 | factory/`model_type` traceback | vendored API gap (config attr, `config_class`, cooperative init) | traceback line in `class_factory.py` → which registry/attribute lookup failed |

Order on any parity failure: identify failing cases → per-layer diff table → matching row → exhausted ⇒ STOP with the table.

## Appendix B — Required report structure

(1) environment block; (2) recon table + 10 answers + F1–F5 validation; (3) key-mapping status; (4) all VERIFY outputs in phase order; (5) parity numbers per case; (6) §12.3/§12.4 outputs; (7) probe losses + YAML diff; (8) files created/modified with one-line purpose (the modified list must be: `lightning_ir/__init__.py`, `lightning_ir/bi_encoder/bi_encoder_tokenizer.py`, and nothing else outside new files); (9) judgment calls not resolved by ground truth (should be none — flag prominently otherwise).

## Out of scope (do not do)

- Merging or cherry-picking from `try_neobert` / `neobertt5` beyond the single marker-token line (§12.2).
- Sequence packing, flash-attention, or performance work; MTEB/BEIR/LongEmbed wiring.
- Training beyond the 20-step probe; Phase 8 items unless explicitly requested after the gate.
- Upstreaming to transformers; opening the PR to lightning-ir `main` (follow-up after all gates, coordinated with the maintainers).
- Touching `mvrT5` itself or rebasing it.
