# NeoBERT — Known Findings & the transformers-v5 Port

Distilled from a long debugging investigation into why `chandar-lab/NeoBERT` produced NaNs
when fine-tuned as a bi-encoder backbone (ColBERT/COL, MVR, etc.) in lightning-ir on the
transformers-v5 stack. **Read this before touching NeoBERT again — it will save days.**

---

## TL;DR

- **NeoBERT loads and runs fine on transformers < 5, and NaNs on transformers 5.x.** That single
  A/B is the whole story. Confirmed: bare `AutoModel`, same input, same torch/xformers →
  **transformers 4.57.6 = finite, 5.12.1 = NaN.**
- **Root cause: transformers v5 breaks NeoBERT's 4.x-era remote code**, in two places, both around
  its **complex RoPE buffer** (`freqs_cis`):
  1. It's a **non-persistent buffer** computed in `__init__`. v5's meta-device loading never
     materializes it (it's not in the checkpoint) → left as **uninitialized garbage / NaN, per
     process** (the source of all the run-to-run flakiness).
  2. It's **`complex64`**. v5's dtype-casting machinery mangles complex buffers → the forward NaNs
     **even when the buffer is finite**.
- **The fix (recommended): vendor NeoBERT's modeling file and port it to v5** — replace the complex
  precomputed RoPE buffer with **real cos/sin computed on the fly in fp32**, and replace
  `xformers.ops.SwiGLU` with a **pure-torch SwiGLU** (same `w12`/`w3` layout so checkpoints load
  unchanged). This keeps everything on your v5 stack and is fully comparable to the other backbones.
- **The lightning-ir integration itself is fine** — loading/inference/save/reload all work (that's
  the `neobertt5` branch). It was never the problem; the model's v5 forward was.

Status at time of writing: the port is **written and numerically reviewed** (the real RoPE matches
the complex reference to machine epsilon); the on-cluster **parity gate + a training probe were
still pending**. Treat "port works" as expected-but-verify until the parity gate passes.

---

## The decisive experiment (do this FIRST for any NeoBERT-v5 NaN)

```python
import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT")
m = AutoModel.from_pretrained("chandar-lab/NeoBERT", trust_remote_code=True).cuda().eval()
enc = tok(["information retrieval systems rank documents by relevance " * 40],
          truncation=True, max_length=256, return_tensors="pt").to("cuda")
print(torch.isfinite(m(**enc).last_hidden_state).all())   # 4.x -> True, 5.x -> False
```

Also inspect buffers: `for n,b in m.named_buffers(): print(n, torch.isfinite(torch.view_as_real(b) if b.is_complex() else b).all())`
→ `freqs_cis` shows `finite=False` on some v5 loads.

---

## Root cause, in detail

NeoBERT's `rotary.py` builds RoPE with `torch.polar(...)` → a **complex64** `freqs_cis` of shape
`(max_length, dim_head/2)`, applied via `view_as_complex` / `view_as_real`. `model.py` registers it
`register_buffer("freqs_cis", ..., persistent=False)` (the code comment even notes non-persistent
buffers aren't saved). Under transformers v5:

- **Meta-device default loading** creates params *and buffers* on `meta`, materialized from the
  checkpoint. `freqs_cis` isn't in the checkpoint → never gets real values → garbage/NaN.
  `low_cpu_mem_usage=False` makes the *buffer* finite again (forces CPU-init), **but the forward
  still NaNs** because of…
- **Complex-dtype casting**: v5's dtype policy / cast machinery corrupts the complex buffer, so even
  a finite `freqs_cis` yields corrupted rotations → NaN, compounding over 28 layers.

Symptoms this explains:
- flaky across runs/nodes (different uninitialized memory each process),
- deterministic within a process (fixed garbage),
- "real text / longer sequences NaN, short/synthetic fine" (incidental — which garbage row got hit),
- the **~8000-magnitude residual activations** we saw were **corruption, not a model property**
  (fp32 at 8000 is ~35 orders of magnitude from overflow; a deterministic fp32 NaN means something
  *emits* NaN, not accumulation).

---

## What it is NOT (ruled out by experiment — don't re-chase these)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| lightning-ir registry / wrapping | ✗ | Adapter wraps it as `ColNeoBERT`; weights load with 0 mismatches vs hub safetensors |
| Padding / attention mask | ✗ | A **single unpadded, fully-real 256-token** sequence NaNs; mask is key-only `(B,1,1,L)` |
| xformers version / SwiGLU | ✗ | Pure-torch SwiGLU + SDPA (zero xformers in forward) still NaNs |
| SDPA vs eager attention | ✗ | Both NaN |
| Precision / fp16 / overflow | ✗ | fp32 **and** bf16 NaN; 8000 can't overflow fp32 |
| Sequence packing needed | ✗ | Tom Aarsen's NeoBERT reranker trains via sentence-transformers on the **padded SDPA** path |
| Embedding OOB / vocab mismatch | ✗ | embedding rows = config vocab = tokenizer len = 30522; max id 29656 < 30522; lookup finite |
| "Massive activations → not fine-tunable" | ✗ (my wrong theory) | The 8000 was corruption; fine on 4.x |
| RoPE buffer alone | ✗ (partial) | `low_cpu_mem_usage=False` fixes `freqs_cis` finiteness but forward still NaN (2nd bug = complex cast) |

---

## The fix — vendored v5 port

Deploy NeoBERT as a **local patched snapshot** (no lightning-ir code change needed; the `neobertt5`
branch already loads remote code from a local dir). Two changed files:

**`rotary.py` — real cos/sin instead of complex `freqs_cis`:**
```python
def compute_cos_sin(seq_len, dim, theta, device, position_ids=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[:dim//2] / dim))
    if position_ids is not None:
        assert position_ids.dim() == 1     # only 1-D (packed/flat) supported
        t = position_ids.to(device=device, dtype=torch.float32)
    else:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)       # (seq_len, dim/2)
    return freqs.cos(), freqs.sin()

def apply_rotary_emb(xq, xk, cos, sin):
    cos, sin = cos[None,:,None,:], sin[None,:,None,:]
    def rotate(x):
        xf = x.float(); x1, x2 = xf[..., 0::2], xf[..., 1::2]     # consecutive-pair (matches view_as_complex!)
        return torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), dim=-1).flatten(-2).type_as(x)
    return rotate(xq), rotate(xk)
```
> The **interleaved** (consecutive-pair) rotation is critical — it matches `view_as_complex` to
> machine epsilon (2.2e-16). A `rotate_half`-style split diverges by ~3.8. This is the classic
> silent-divergence trap in RoPE ports; get it right.

**`model.py` changes (surgical — everything else identical to the hub file):**
- `from xformers.ops import SwiGLU` → a pure-torch `SwiGLU` with `w12` (fused, `2*hidden`) and `w3`
  layout: `x1,x2 = w12(x).chunk(2,-1); return w3(F.silu(x1) * x2)`. Same weight names → checkpoint
  loads unchanged; kills the xformers dependency entirely.
- Remove the `freqs_cis` `register_buffer`. In `NeoBERT.forward`, after the embedding, compute
  `cos, sin = compute_cos_sin(x.shape[1], config.dim_head, 10000.0, x.device, position_ids)` and
  thread `cos, sin` into each `EncoderBlock` / `_att_block` (replacing the `freqs_cis` arg).
- SDPA branch: guard the mask — `attn_mask=attention_mask.bool() if attention_mask is not None else None`.
- `NeoBERTForSequenceClassification`: `getattr(self.config, "use_return_dict", True)` (v5 deprecates the attr).

Design principle: **no buffer at all** (nothing for meta-loading to leave uninitialized) and
**real fp32 cos/sin** (no complex dtype for the cast machinery to mangle). Both v5 bugs die at once.

### Deployment
```bash
# 1. Build the patched snapshot: copy config/weights/tokenizer from the hub cache (NOT the .py)
python - << 'PY'
import shutil, pathlib
from huggingface_hub import snapshot_download
src = pathlib.Path(snapshot_download("chandar-lab/NeoBERT")); dst = pathlib.Path("neobert-v5"); dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.suffix in {".json", ".safetensors", ".txt"}: shutil.copy(f, dst / f.name)
PY
# 2. Drop the patched rotary.py + model.py into neobert-v5/
# 3. Use it: model_name_or_path: /abs/path/to/neobert-v5   (lightning-ir loads the local remote code)
```

### Acceptance gate (do this before trusting any training)
Parity check — same padded 3-doc batch through **hub code @ transformers 4.x** vs **patched @ 5.x**;
save `last_hidden_state` from the 4.x reference, reload and diff on 5.x. Pass criteria:
- `finite = True`,
- `max abs diff` ~**1e-4 – 1e-3** (dominated by torch-vs-xformers SwiGLU kernel + SDPA-backend rounding),
- `port max|h| ≈ ref max|h|` at a **sane magnitude (NOT ~8000)** — confirming the blow-up was corruption.

Then a ~20-step COL training probe → loss must be finite/decreasing before the full run.

---

## Alternative (fallback) — pin transformers < 5

Running NeoBERT in a `transformers>=4.48,<5` env works out of the box (recipe is portable). But
lightning-ir's `neobertt5`/`mvrt5` branch uses v5 APIs, so NeoBERT would run on a **different stack**
than your other backbones — a comparability confound for the paper. Prefer the port.

---

## The lightning-ir integration fixes (`neobertt5` branch) — separate from the v5 forward bug

These are correct and needed for **load / inference / save / reload** of any remote-code backbone;
they were NOT the NaN cause:
- Remote-code resolution: `AutoConfig`/`auto_map` fallbacks for config, model class, and tokenizer
  when the backbone type isn't in the transformers mappings.
- `_init_weights`: initialize the projection head (else it's uninitialized garbage → inf scores) AND
  **skip modules whose params are already `_is_hf_initialized`** (else reload clobbers loaded weights).
- `get/set_input_embeddings` → `self.encoder` for neobert (so `resize_token_embeddings` works for
  MVR's `[VIE*]` viewer tokens). *(Not needed for COL, which adds no tokens.)*
- `save_pretrained` override: copies the backbone's dynamic-module `.py` into the checkpoint so a
  separate eval job can reload it. *(The v5 port makes this less relevant — the vendored dir is
  self-contained.)*
- `trust_remote_code=True` default in the tokenizer fallback (so unattended Slurm jobs don't hang on
  the interactive `[y/N]` prompt).
- `neobert` in `ADD_MARKER_TOKEN_MAPPING`.

---

## Environment / ops gotchas

- **`~/.local` shadowing**: pip in the Slurm container defaults to the persistent `~/.local`
  (site-packages not writable), which can be polluted/broken (`~ransformers`, NCCL-mismatched torch).
  Always run in a **venv** with `export PYTHONNOUSERSITE=1`; never `pip install --user`.
- **Slurm script must**: `source venv3/bin/activate`, `PYTHONNOUSERSITE=1`, install lightning-ir with
  `--no-deps` (don't disturb the pinned torch/xformers), and NOT `pip install torchvision`/`xformers`
  unpinned (that's what produced a broken torch 2.11 / xformers 0.0.35 stack).
- **After the port, xformers is no longer needed** (the vendored `model.py` doesn't import it).
- **`--gres=gpu:hopper:1`** (80 GB) fits COL at `train_batch_size: 16`; Ampere (40 GB) OOMs → drop to
  batch 8 / accumulate 8 (keep effective batch 64 for comparability).
- Known-working env used during the investigation: torch 2.7.1+cu128, transformers 5.12.1
  (venv3). The A/B reference used transformers 4.57.6 (venv_t4).

---

## Debugging methodology lessons (why this took so long)

- **Reproduce the EXACT failing condition first.** The failure was real-MS-MARCO training; we spent
  days on synthetic/isolated forwards that behaved differently (flaky/finite) and drew false
  conclusions from them (e.g. "env fixed it", "massive activations").
- **For a NaN, measure directly and early:** per-layer activation magnitude on REAL data,
  `named_buffers()` finiteness, and — for remote code under a major transformers bump — the
  **A/B on transformers<5**. That last one would have found this in one step.
- **A flaky, per-process NaN = uninitialized memory / buffer.** A **deterministic fp32 NaN =
  something emits NaN** (garbage buffer, div-by-~0), not gradual overflow. 8000 ≪ fp32 inf.
- **"forward runs" ≠ "loaded correctly"** under a major-version jump — checksum weights, check buffers.
- Once fixed, **upstream it**: transformers#37015 ("Add NeoBERT") is open with no PRs; the RetroMAE
  fork vendors the same buggy Feb-2025 code. A hub discussion + the v4/v5 A/B repro + this port is
  what the ecosystem is missing.

---

## F2b — the transformers-v5 guarded-init trap (confirmed during the vendored port)

**One-line rule:** in transformers v5, `_init_weights` must go through the *guarded*
`transformers.initialization` functions (`init.normal_`, `init.uniform_`, `init.zeros_`,
`init.ones_`, …) operating **on the Parameter** — never raw `module.weight.data.uniform_(...)`.

**Why it bites (the exact mechanism):** v5's `from_pretrained` no longer relies on a separate
"which modules were loaded" pass to gate re-init. Instead it *always* runs `_init_weights` over the
whole model after loading, but wraps that pass in `@init.guard_torch_init_functions()`
(`initialize_weights`, `modeling_utils.py`). The guarded init functions check
`_is_hf_initialized` **on the tensor** and no-op on params that were just loaded from the checkpoint.
Loaded params carry that flag; freshly-created ones don't. So:

- `init.uniform_(module.weight, ...)` → guarded → **skips** loaded weights, initializes only the
  genuinely-missing ones (e.g. a from-scratch Col projection). Correct.
- `module.weight.data.uniform_(...)` → operates on the detached `.data` view, **bypasses the guard**
  → re-randomizes **every** weight *after* it was loaded. The model silently becomes random.

**Why it's nasty (same class as the tie bug):** it corrupts weights while `strict` load still
reports **0 missing / 0 unexpected** — the keys matched, the values got clobbered afterwards. A
finite-but-wrong model passes every structural check (loads, forwards, finite loss). Only a
save→load roundtrip *value* comparison catches it. This is almost certainly the same family of
"forward runs ≠ loaded correctly" failure the original remote code hit under v5.

**Canary (cheap, catches both this and the tie-machinery bug):** after a full save→load roundtrip,
assert every tensor equals what was saved, **and** `not torch.equal(decoder.weight,
model.encoder.weight)` (an untied decoder must stay untied). Both bug classes are silent under
strict load; this one roundtrip assertion is the guard. Encoded in
`scripts/neobert/convert_checkpoint.py` and `tests/test_models/test_neobert_unit.py`.

**Upstream relevance:** transformers#37015 + the v4/v5 A/B repro (F6) + *this* init-API trap is
exactly the kind of "here's why 4.x-era custom-code models break on v5" writeup maintainers want —
F6 is the RoPE-buffer half, F2b is the init-guard half.
