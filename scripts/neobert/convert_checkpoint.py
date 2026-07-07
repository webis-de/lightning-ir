"""Convert the pristine ``chandar-lab/NeoBERT`` hub checkpoint into a vendored, self-contained
lightning-ir checkpoint that loads via ``AutoModel.from_pretrained`` with no remote code.

Origin snapshot: ``chandar-lab/NeoBERT`` revision ``5424c8efeea6491b151d62dee55a752165407430``.

Usage (main env, CPU is fine):
    python scripts/neobert/convert_checkpoint.py \
        --src third_party/neobert_original \
        --dst checkpoints/neobert-vendored
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

from lightning_ir.models.backbones.neobert import NeoBERTConfig, NeoBERTForMaskedLM

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("third_party/neobert_original"))
    ap.add_argument("--dst", type=Path, default=Path("checkpoints/neobert-vendored"))
    args = ap.parse_args()

    original = load_file(args.src / "model.safetensors")
    model = NeoBERTForMaskedLM(NeoBERTConfig())

    orig_keys = set(original)
    vendored_keys = set(model.state_dict())

    only_in_original = sorted(orig_keys - vendored_keys)
    only_in_vendored = sorted(vendored_keys - orig_keys)
    print(f"[keys] original={len(orig_keys)} vendored={len(vendored_keys)}")
    print(f"[keys] only in ORIGINAL ({len(only_in_original)}):")
    for k in only_in_original:
        print("   ", k, tuple(original[k].shape))
    print(f"[keys] only in VENDORED ({len(only_in_vendored)}):")
    for k in only_in_vendored:
        print("   ", k, tuple(model.state_dict()[k].shape))

    # Build the converted state dict. Mapping is identity (attribute hierarchy preserved).
    ORIG_TO_VENDORED: dict[str, str] = {}  # intentionally empty: identity mapping
    converted = {ORIG_TO_VENDORED.get(k, k): v for k, v in original.items()}

    # Shape check for every shared key (a name match with a shape mismatch = wrong arch constant).
    vsd = model.state_dict()
    shape_mismatches = [
        (k, tuple(converted[k].shape), tuple(vsd[k].shape))
        for k in converted
        if k in vsd and converted[k].shape != vsd[k].shape
    ]
    if shape_mismatches:
        raise SystemExit(f"STOP: shape mismatches (wrong architecture constant): {shape_mismatches}")

    missing, unexpected = model.load_state_dict(converted, strict=False)
    print(f"[load] missing={list(missing)} unexpected={list(unexpected)}")
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"

    # The MLM decoder is NOT tied to the input embedding (verified in recon; enforce here).
    assert not torch.equal(
        model.decoder.weight.data, model.model.encoder.weight.data
    ), "STOP: decoder.weight became tied to model.encoder.weight (tie_word_embeddings leaked True)."
    print("[tie] decoder.weight is untied from model.encoder.weight: OK")

    # Preserve original dtype.
    orig_dtype = next(iter(original.values())).dtype
    model = model.to(orig_dtype)

    args.dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)

    for name in TOKENIZER_FILES:
        f = args.src / name
        if f.exists():
            shutil.copy(f, args.dst / name)
    # Never copy model.py / rotary.py — the vendored dir must be self-contained (no remote code).
    for forbidden in ("model.py", "rotary.py"):
        assert not (args.dst / forbidden).exists(), f"STOP: {forbidden} must not be in the vendored checkpoint."

    print(f"[save] wrote vendored checkpoint to {args.dst} (dtype={orig_dtype})")

    # Self-check: the checkpoint must reload through from_pretrained with weights intact (guards
    # against the F2 clobber — _init_weights re-initialising loaded params on the reload path).
    reloaded = NeoBERTForMaskedLM.from_pretrained(args.dst)
    rsd = reloaded.state_dict()
    n_ok = sum(1 for k, v in converted.items() if k in rsd and torch.equal(rsd[k].to(v.dtype), v))
    assert n_ok == len(converted), f"STOP: {len(converted) - n_ok}/{len(converted)} tensors clobbered on reload (F2b)."
    # Canary for BOTH the F2b guarded-init trap and the tie-machinery bug (both corrupt silently
    # while strict load reports 0 missing / 0 unexpected): the decoder must stay untied post-roundtrip.
    assert not torch.equal(
        reloaded.decoder.weight, reloaded.model.encoder.weight
    ), "STOP: decoder tied to input embedding after save->load roundtrip (tie-machinery/F2b canary)."
    print(f"[reload] from_pretrained restored all {n_ok}/{len(converted)} tensors, decoder still untied: OK")


if __name__ == "__main__":
    main()
