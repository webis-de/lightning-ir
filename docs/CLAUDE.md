# CLAUDE.md — lightning-ir, branch `neobert-backbone`

## What this branch is

Vendoring NeoBERT (`chandar-lab/NeoBERT`) as a native lightning-ir backbone on the
transformers-v5 stack. The authoritative task definition is
**`docs/neobert_vendoring_plan.md`** — read it fully before doing anything, and follow
its phase order, VERIFY/GT/STOP protocol, and per-phase Definitions of Done exactly.
Supporting material:

- `docs/neobert_known_findings.md` — debugging post-mortem; findings F1–F10 are
  **confirmed facts**, not hypotheses. Do not re-litigate them.
- `third_party/neobert_v5_patched/{model.py,rotary.py}` — the numerically reviewed
  v5 port. Phase 2 **starts from these files**; keep every numerical decision intact
  (interleaved RoPE, no RoPE buffer of any kind, SwiGLU `w12`/`w3` layout with the
  `multiple_of = 8` intermediate sizing).
- `third_party/neobert_original/` — pristine hub snapshot, read-only, gitignored;
  ground-truth reference only.

## Hard rules (non-negotiable)

1. **No dependency changes.** Never edit `pyproject.toml` dependencies. `xformers`
   and `flash_attn` exist only in the throwaway reference venv (spec §11.1), never
   in the main env.
2. **No `trust_remote_code`** anywhere except `scripts/neobert/dump_reference_outputs.py`.
3. **Do not modify `lightning_ir/base/`** (`class_factory.py`, `model.py`,
   `tokenizer.py`, ...). The design premise is that Auto-class registration alone
   suffices. If something seems to require touching `base/`, STOP and report the
   traceback — it indicates a bug in the vendored model, not a missing factory feature.
   Allowed edits outside new files: the registration hook in `lightning_ir/__init__.py`
   and the one-line `ADD_MARKER_TOKEN_MAPPING` entry in
   `lightning_ir/bi_encoder/bi_encoder_tokenizer.py`. Nothing else.
4. **Do not change model numerics** relative to the patched port (computation order,
   norm placement, eps, dtype casts, RoPE convention).
5. **Never loosen a tolerance, skip a VERIFY, or claim a checkbox without pasted
   output.** When a gate fails, use the spec's Appendix A decision tree, then STOP.
6. **Do not merge or cherry-pick from `try_neobert` / `neobertt5`** (superseded
   remote-code attempts) beyond what the spec explicitly lists.
7. **Do not touch the `mvrT5`/`mvrt5` branch** or rebase anything onto it.

## Environment

- Always work in the project venv with `PYTHONNOUSERSITE=1` exported. Never
  `pip install --user` (a polluted `~/.local` has broken this setup before).
- lightning-ir is installed editable: `pip install -e . --no-deps`. Main env is
  transformers 5.x + torch; verify with the spec's Phase −1 block before starting.
- GPU work (Phase 7 probe) runs on the cluster: hopper (80 GB) fits Col at
  `train_batch_size: 16`; on Ampere (40 GB) use 8 with doubled
  `accumulate_grad_batches` (keep effective batch 64).

## Conventions

- New code lives under the paths defined in spec §5; mirror existing lightning-ir
  style (type hints, short docstrings, no dead code).
- Every vendored file gets a header: origin (`chandar-lab/NeoBERT` + snapshot
  revision), MIT license notice, one-line list of modifications.
- Tests follow `tests/conftest.py` conventions; anything needing the parity fixture,
  the converted checkpoint, or a GPU must skip cleanly so CI stays green.
- Commit after every green phase, one phase per commit, message prefixed with the
  phase (e.g. `phase 3: checkpoint conversion, strict load clean`). Never commit
  weights, the hub snapshot, `checkpoints/`, or large fixtures.
- End every phase with the report elements the spec's Appendix B requires; when in
  doubt between proceeding and asking — ask.
