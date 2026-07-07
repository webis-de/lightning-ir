# NeoBERT vendoring — Phase −1 & Phase 0 recon report

Structure follows spec Appendix B. Sections (5)/(6)/(7) belong to later phases and are
marked **pending**. All GT line references point at `third_party/neobert_original/`
(hub snapshot, revision `5424c8efeea6491b151d62dee55a752165407430`).

> **ACTION ITEM (blocking, hygiene): `third_party/neobert_original/` is NOT gitignored.**
> It holds a 980 MB `model.safetensors`. Spec §5 and CLAUDE.md require the snapshot/weights
> to be gitignored. Recommend adding `third_party/neobert_original/` to `.gitignore` before
> anything is staged. Not done yet (awaiting review; no files written).

---

## (1) Environment block

```
sys.version    : 3.12.13 (main, Mar 10 2026) [Clang 17.0.0]
torch          : 2.10.0   cuda=None   cuda.is_available()=False   (local macOS CPU box)
transformers   : 5.12.1   (>= 5.0.0  OK)
lightning_ir   : 0.0.6    /Users/wilhelmpertsch/dev/lightning-ir/lightning_ir  (editable, in-tree OK)
branch         : neobert-backbone
merge-base     : "based on mvrt5: OK"
torch.nn.RMSNorm present : True
```

**Notes / not-a-STOP:** this is the local dev box (CPU-only; `cuda=None`). Phase −1 DoD requires
only transformers v5.x + editable install + mvrt5 ancestry — all satisfied. The cluster env
(torch 2.7.1+cu128) is where the Phase 7 GPU probe runs. `torch.nn.RMSNorm` exists, so §8.5 can
use it directly (no local RMSNorm fallback needed in the main env).

---

## (2) Recon table + 10 GT answers + F1–F5 validation

### Class / module table (original `model.py`)

| Class | Ctor args | Submodule attrs (nesting) | External imports |
|---|---|---|---|
| `NeoBERTConfig` (`model.py:68`) | `hidden_size=768, num_hidden_layers=28, num_attention_heads=12, intermediate_size=3072, embedding_init_range=0.02, decoder_init_range=0.02, norm_eps=1e-6, vocab_size=30522, pad_token_id=0, max_length=1024, **kwargs` | derives `dim_head = hidden//heads` | `PretrainedConfig` |
| `EncoderBlock` (`model.py:104`) | `(config)` | `qkv` (Linear 768→2304, bias=False), `wo` (Linear 768→768, bias=False), `ffn` (`SwiGLU`), `attention_norm` (RMSNorm), `ffn_norm` (RMSNorm) | `xformers.ops.SwiGLU`, `scaled_dot_product_attention`, `flash_attn_varlen_func` (optional) |
| `NeoBERTPreTrainedModel` (`model.py:200`) | — | `config_class`, `base_model_prefix="model"`, `_supports_cache_class=True`, `_init_weights` | `PreTrainedModel` |
| `NeoBERT` (`model.py:212`) | `(config)` | `encoder` (Embedding, padding_idx=0), `freqs_cis` (non-persistent complex buffer), `transformer_encoder` (ModuleList[28×EncoderBlock]), `layer_norm` (RMSNorm) | `precompute_freqs_cis`, `apply_rotary_emb`, `BaseModelOutput` |
| `NeoBERTLMHead` (`model.py:300`) | `(config)` | `model` (NeoBERT), `decoder` (Linear 768→30522, bias=True) | `MaskedLMOutput` |
| `NeoBERTForSequenceClassification` (`model.py:343`) | `(config)` | `model`, `dense`, `dropout`, `classifier` | `SequenceClassifierOutput`, losses |
| `DataCollatorWithPacking` (`model.py:35`) | — packing collator; **out of scope** (flash/packed path) | — | `DataCollatorForLanguageModeling` |

### 10 questions (each with GT line ref)

1. **QKV fused or separate? attr?** — **Fused, single Linear.** `self.qkv = nn.Linear(hidden, hidden*3, bias=False)` (`model.py:113`). Consumed at `model.py:159`: `self.qkv(x).view(B, S, H, dim_head*3).chunk(3, axis=-1)` → order **q, k, v** (chunk on the last, per-head-grouped dim, *after* the `view`). Output projection is `self.wo` (Linear 768→768, bias=False, `model.py:114`), applied `model.py:197`. Keys: `...qkv.weight (2304,768)`, `...wo.weight (768,768)`.
2. **FFN module / import / ctor?** — `from xformers.ops import SwiGLU` (`model.py:12`); constructed `SwiGLU(hidden_size, intermediate_size, hidden_size, bias=False)` (`model.py:120`) where `intermediate_size` is `int(2*3072/3)=2048` rounded up to a multiple of 8 = **2048** (`model.py:117-119`). **Packed `w12`/`w3` layout** confirmed by keys: `ffn.w12.weight (4096,768)` (=2×2048) and `ffn.w3.weight (768,2048)`. → **F8 validated.**
3. **Norm type / pre-post / eps / final norm?** — `torch.nn.RMSNorm`, **pre-norm**. Per block (`model.py:136-144`): `x = x + attn(attention_norm(x)); x = x + ffn(ffn_norm(x))`. Norms at `model.py:123-124`; final `self.layer_norm` (RMSNorm) after the last layer at `model.py:230` / applied `model.py:290`. **eps = `config.norm_eps`**; **published `config.json` sets `1e-5`** (the `model.py` default `1e-6` is overridden on load — see discrepancy in §9). RMSNorm is weight-only (no bias) — keys show only `*.weight` for all norms.
4. **token_type / absolute pos / token-emb attr?** — **No token-type, no absolute-position embeddings — RoPE only.** Token embedding = `self.encoder = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)` (`model.py:220`; key `model.encoder.weight (30522,768)`). The original implements **no** `get/set_input_embeddings`. → **F1 validated** (unusual `encoder` attr name).
5. **Does `forward` use `token_type_ids`?** — **No.** `NeoBERT.forward` (`model.py:235-245`) has no `token_type_ids` parameter; `**kwargs` silently swallows it. The tokenizer does not emit it either (see §6.4). The vendored port must still *accept* `token_type_ids=None` gracefully.
6. **Padding-mask consumption / eager path?** — `(B,L)` mask → `.unsqueeze(1).unsqueeze(1).repeat(1, H, L, 1)` → **`(B,H,L,L)` bool** (`model.py:254-255`), key-only (each query sees the same key mask). SDPA branch (`model.py:187-195`): `attn_mask=attention_mask.bool()` (True=attend). **This SDPA branch is the unpadded/eager path to reproduce.** The `output_attentions` branch (`model.py:180-186`) does a *multiplicative* pre-softmax mask (not `-inf`) — technically lossy; relevant only for Phase-4 output_attentions handling. The packing branch (`cu_seqlens`, `model.py:167-178`) is flash-attn — out of scope. **NOTE:** original `model.py:193` calls `attention_mask.bool()` with no `None` guard; the patched port added `... if attention_mask is not None else None`.
7. **Dropout?** — **None in the base model** (no attention/hidden dropout, no config fields). Only `NeoBERTForSequenceClassification` uses `classifier_dropout` (default 0.1, `model.py:352`). → attention `dropout_p = 0.0`.
8. **RoPE fn / where / convention / origin+dtype / layout?** — `precompute_freqs_cis` (`rotary.py:7`) builds a **complex64** `freqs_cis` via `torch.polar` (`rotary.py:27`), registered as a **non-persistent buffer** (`model.py:224`). `apply_rotary_emb` (`rotary.py:35`) uses `view_as_complex(x.reshape(...,-1,2))` — **interleaved consecutive-pair** rotation. Applied to `xq,xk` **before** attention (`model.py:161`), tensor layout **`(B, S, H, dim_head)`** (heads on dim 2). `theta=10000.0` (`rotary.py:7` default), freqs in fp32. → **F6 validated** (the exact non-persistent complex64 buffer) and **F7 validated** (consecutive-pair == `view_as_complex`). The patched port's `[B,S,H,D]` broadcast (`cos[None,:,None,:]`) matches this layout — so §8.3's `[B,H,S,D]` skeleton yields to the patched files, per §8.2.
9. **MLM decoder tied?** — **Not tied.** `NeoBERTLMHead.decoder = nn.Linear(hidden, vocab)` (`model.py:309`), separate from `model.encoder`; no tie logic, `config` has no `tie_word_embeddings`. Checkpoint proves it: `decoder.weight` and `model.encoder.weight` are **not equal** (`max|Δ| = 0.984`), and `decoder.bias (30522,)` is present and nonzero. → separate decoder weights present; do **not** tie.
10. **Head classes & `base_model_prefix`?** — `NeoBERT` (base; `AutoModel`), `NeoBERTLMHead` (`AutoModelForMaskedLM`; `config.architectures = ["NeoBERTLMHead"]`, i.e. the checkpoint *is* the MLM head), `NeoBERTForSequenceClassification`. `base_model_prefix = "model"` (`model.py:202`), `_supports_cache_class = True` (`model.py:203`). Checkpoint nesting: base under `model.` prefix; MLM decoder at top level `decoder.*`.

### Original `_init_weights` vs the transformers per-module convention (F2/F3)

Original (`model.py:205-209`): generic over two types —
`nn.Linear → weight.uniform_(-decoder_init_range, +decoder_init_range)`;
`nn.Embedding → weight.uniform_(-embedding_init_range, +embedding_init_range)`.
Differences from the §8.7 recommended convention:
- **`uniform_(±range)`, not `normal_(0, std)`** (ranges both 0.02).
- **No `padding_idx` zeroing** for the embedding.
- **No RMSNorm/LayerNorm handling** (relies on `nn.RMSNorm` default weight=1).
- **No Linear-bias zeroing** (encoder Linears are all bias-free; only `decoder.bias` exists).
- It *is* the standard per-module signature (`_init_weights(self, module)`), so transformers'
  `_is_hf_initialized` gate applies automatically.
- **Being generic over `nn.Linear` means it already initializes lightning-ir-added Linear heads
  (Col projection)** — so, unlike modernbert, **NeoBERT needs no special-case in `base/model.py`.**

### F1–F5 validation

| Finding | Verdict | Evidence |
|---|---|---|
| **F1** token emb `self.encoder`, no get/set | **VALIDATED** | `model.py:220`; no `get_input_embeddings`/`set_input_embeddings` in original. |
| **F2** reload-clobber risk / init gating | **VALIDATED (mechanism)** | `base/model.py:97-102` shows the lightning-ir `_init_weights` override + `PreTrainedModel._init_weights` re-init for modernbert. Vendored generic per-module `_init_weights` + `post_init()` (respecting `_is_hf_initialized`) avoids clobber; §12.4-test-3 guards it. |
| **F3** uninitialized heads / no neobert special-case | **VALIDATED** | `base/model.py:99-102` is a modernbert-only `_init_weights` special-case. NeoBERT's init is generic over `nn.Linear`, so the Col projection is initialized → no `"neobert"` branch needed. |
| **F4** BERT-family markers `[CLS]`/`[SEP]` | **VALIDATED** | tokenizer is BERT-family, `[CLS]=101`, `[SEP]=102` (§6.4). Current `ADD_MARKER_TOKEN_MAPPING` has `bert` + `modernbert` with the identical pattern; the neobert line (§12.2) applies verbatim. |
| **F5** tokenizer KeyError → trust_remote_code | **VALIDATED** | Unregistered `neobert` → AutoTokenizer can't map config→tokenizer, falls back to the slow `BertTokenizer` via the `tokenizer_class` string and emits a noisy `model of type neobert` warning (§6.4). Registration with `fast_tokenizer_class=BertTokenizerFast` (§12.1) fixes resolution. |

**Correction to an F4 sub-implication:** the NeoBERT tokenizer does **not** emit `token_type_ids`
(`model_input_names = ["input_ids","attention_mask"]`), so the "BERT tokenizer emits token_type_ids
regardless" assumption is false here. Harmless — the vendored forward accepts/ignores it anyway.

**F6/F7/F8** (post-mortem facts) cross-checked against the original source: all confirmed above
(F6 @ `model.py:224`+`rotary.py:27`; F7 @ `rotary.py:56`; F8 @ keys + `model.py:120`).

---

## (3) Key-mapping status

- Keys file written: `scripts/neobert/reference_state_dict_keys.txt` (172 keys, all `torch.float32`).
- **172 = 4 non-layer + 6 × 28 layers.** Non-layer: `decoder.bias (30522,)`, `decoder.weight (30522,768)`,
  `model.encoder.weight (30522,768)`, `model.layer_norm.weight (768,)`. Per layer:
  `attention_norm.weight (768,)`, `ffn.w12.weight (4096,768)`, `ffn.w3.weight (768,2048)`,
  `ffn_norm.weight (768,)`, `qkv.weight (2304,768)`, `wo.weight (768,768)`.
- **Expected mapping = identity / empty.** The patched port keeps the exact attribute hierarchy
  (`encoder`, `transformer_encoder.N.{qkv,wo,ffn.w12,ffn.w3,attention_norm,ffn_norm}`, `layer_norm`,
  MLM `decoder`), so a strict load should show zero missing / zero unexpected once the RoPE buffer
  is gone (no `freqs_cis` in the checkpoint anyway — it was non-persistent). Confirmed in Phase 3.

---

## (4) VERIFY outputs (phase order)

**§6.4 state dict — first 10 + summary:**
```
TOTAL KEYS: 172        DTYPES: {'torch.float32'}
decoder.bias                                  (30522,)      float32
decoder.weight                                (30522, 768)  float32
model.encoder.weight                          (30522, 768)  float32
model.layer_norm.weight                       (768,)        float32
model.transformer_encoder.0.attention_norm.weight (768,)    float32
model.transformer_encoder.0.ffn.w12.weight    (4096, 768)   float32
model.transformer_encoder.0.ffn.w3.weight     (768, 2048)   float32
model.transformer_encoder.0.ffn_norm.weight   (768,)        float32
model.transformer_encoder.0.qkv.weight        (2304, 768)   float32
model.transformer_encoder.0.wo.weight         (768, 768)    float32
```

**Original `config.json` (full):**
```json
{
  "architectures": ["NeoBERTLMHead"],
  "auto_map": {
    "AutoConfig": "model.NeoBERTConfig",
    "AutoModel": "model.NeoBERT",
    "AutoModelForMaskedLM": "model.NeoBERTLMHead",
    "AutoModelForSequenceClassification": "model.NeoBERTForSequenceClassification"
  },
  "classifier_init_range": 0.02,
  "decoder_init_range": 0.02,
  "dim_head": 64,
  "embedding_init_range": 0.02,
  "hidden_size": 768,
  "intermediate_size": 3072,
  "kwargs": {"classifier_init_range": 0.02, "pretrained_model_name_or_path": "google-bert/bert-base-uncased", "trust_remote_code": true},
  "max_length": 4096,
  "model_type": "neobert",
  "norm_eps": 1e-05,
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "pad_token_id": 0,
  "pretrained_model_name_or_path": "google-bert/bert-base-uncased",
  "torch_dtype": "float32",
  "transformers_version": "4.48.2",
  "trust_remote_code": true,
  "vocab_size": 30522
}
```

**§6.4 tokenizer VERIFY (raw):**
```
CLASS: BertTokenizer          padding_side: right
enc("hello world") -> {'input_ids': [101, 7592, 2088, 102], 'attention_mask': [1,1,1,1]}   # keys: input_ids, attention_mask only
pad/cls/sep/mask/unk ids: 0 101 102 103 100     vocab len: 30522
batch padding row0 attn: [1,1,1,1,1,0,0,0,0,0,0,0,0,0]   -> right padding confirmed
with trust_remote_code=False, use_fast=True -> still BertTokenizer (slow); token_type_ids NOT emitted
```
- Loads **without remote code** (clean with explicit `trust_remote_code=False`; no `[y/N]` hang).
- Resolves to **slow `BertTokenizer`** pre-registration (fast class requires the §12.1 registration).
- GT registration classes: `slow_tokenizer_class=BertTokenizer`, `fast_tokenizer_class=BertTokenizerFast`.

**Decoder-tying VERIFY (§6.3 Q9):**
```
torch.equal(decoder.weight, model.encoder.weight): False    max|Δ|: 0.984    decoder.bias nonzero: True
```

---

## (5) Parity numbers — **pending** (Phase 5).
## (6) §12.3 / §12.4 outputs — **pending** (Phase 6).
## (7) Probe losses + YAML diff — **pending** (Phase 7).

---

## (8) Files created / modified so far

Created (recon artifacts only — no model/library code written):
- `third_party/neobert_original/` — pristine hub snapshot (rev `5424c8ef…`), read-only GT reference. **NOT gitignored yet — action item.**
- `scripts/neobert/reference_state_dict_keys.txt` — GT key/shape/dtype dump (172 keys).
- `docs/neobert_recon_report.md` — this report.

No edits to `lightning_ir/` or `pyproject.toml`. The eventual "modified outside new files" list
(spec §8 constraint) remains `lightning_ir/__init__.py` + `lightning_ir/bi_encoder/bi_encoder_tokenizer.py` only.

**File-layout deviation to record (spec §5):** repo tests live in `tests/test_models/*.py`
(`test_col.py`, `test_mvr.py`, …) gated by `@pytest.mark.model` + `--run-models`
(`tests/conftest.py:50,64,66`); there is no `tests/test_backbones/`. Proposal: place the new tests
under `tests/test_models/` following the existing marker/gating convention rather than spec §5's
`tests/test_backbones/` — to confirm with you in Phase 3/5.

---

## (9) Judgment calls not resolved by ground truth (flagged for your review)

1. **Config defaults ≠ published architecture.** `model.py`'s `NeoBERTConfig.__init__` defaults are
   `norm_eps=1e-6`, `max_length=1024`, but `config.json` ships `norm_eps=1e-5`, `max_length=4096`.
   Loading from the checkpoint overrides them (numerics stay correct), but spec §7 DoD wants
   bare `NeoBERTConfig()` to reproduce the published arch. **Recommendation:** in the vendored
   `NeoBERTConfig`, set defaults to the `config.json` values (`norm_eps=1e-5`, `max_length=4096`,
   plus `dim_head=64`, the `*_init_range=0.02` fields). Also drop `auto_map`/`trust_remote_code`/
   `kwargs`/`pretrained_model_name_or_path` from what the vendored config emits.
2. **`_init_weights` distribution: `uniform_` (GT) vs `normal_` (§8.7 skeleton).** The original uses
   `uniform_(±0.02)` with `embedding_init_range`/`decoder_init_range`; the §8.7 code skeleton shows
   `normal_(0, std)`. Spec §8.7 text says "match the original init distribution where the original
   specifies one (GT)". **Recommendation:** follow GT — implement `uniform_` with the two ranges,
   *and* add the padding_idx zeroing + RMSNorm(weight=1) handling the original omits (harmless for
   parity, correct for from-scratch Col/MVR heads). Flagging because it contradicts the literal
   skeleton. Please confirm uniform-with-ranges is the intended target.
3. **`token_type_ids` not emitted by the tokenizer** (correction to F4 sub-implication). No action
   beyond ensuring the vendored `forward` accepts/ignores `token_type_ids=None`.
4. **`third_party/neobert_original/` gitignore** (repeated from top): needs a `.gitignore` entry
   before staging to avoid committing the 980 MB weights. Awaiting your go-ahead.

No other unresolved judgment calls.
