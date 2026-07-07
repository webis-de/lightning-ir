"""Dump reference outputs from the ORIGINAL NeoBERT remote code for the parity gate.

RUN ONLY in the pinned reference venv (spec 11.1): ``transformers==4.57.6`` + ``xformers``, the
proven-finite reference stack. This is the single script permitted to use ``trust_remote_code=True``.
CPU, fp32 — no GPU needed.

    python -m venv /tmp/neobert_ref && source /tmp/neobert_ref/bin/activate
    export PYTHONNOUSERSITE=1
    pip install "transformers==4.57.6" torch xformers safetensors
    python scripts/neobert/dump_reference_outputs.py \
        --src third_party/neobert_original \
        --out tests/test_models/fixtures/reference_outputs.pt

Then copy ``reference_outputs.pt`` back to the main env and run ``test_neobert_parity.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

SNAPSHOT_REVISION = "5424c8efeea6491b151d62dee55a752165407430"


def build_inputs(tokenizer):
    base = "information retrieval systems rank documents by relevance "
    short = "information retrieval systems rank documents"  # ~7-10 tokens
    medium = base * 30  # ~200 tokens
    long = base * 220  # ~1500 tokens, exceeds 512 to exercise the RoPE range
    padded_batch = [  # 4 clearly-different lengths -> right padding
        "short query",
        base * 5,
        base * 20,
        base * 60,
    ]
    cases = {}
    cases["short"] = tokenizer(short, return_tensors="pt")
    cases["medium"] = tokenizer(medium, truncation=True, max_length=256, return_tensors="pt")
    cases["long"] = tokenizer(long, truncation=True, max_length=2048, return_tensors="pt")
    cases["padded_batch"] = tokenizer(padded_batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("third_party/neobert_original"))
    ap.add_argument("--out", type=Path, default=Path("tests/test_models/fixtures/reference_outputs.pt"))
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.src)
    model = AutoModel.from_pretrained(args.src, trust_remote_code=True, torch_dtype=torch.float32).eval()

    cases = build_inputs(tokenizer)

    # VERIFY right padding on the batched case.
    pb = cases["padded_batch"]
    assert (pb["attention_mask"][:, 0] == 1).all(), "expected right padding (first column all attended)"
    print("[verify] padded_batch attention_mask row lengths:", pb["attention_mask"].sum(-1).tolist())

    results = {}
    has_hidden_states = True
    for name, enc in cases.items():
        with torch.no_grad():
            try:
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states
            except TypeError:
                out = model(**enc)
                hs = None
                has_hidden_states = False
        results[name] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "last_hidden_state": out.last_hidden_state,
            "hidden_states": None if hs is None else [h for h in hs],
        }
        n_hs = 0 if hs is None else len(hs)
        print(
            f"[dump] {name}: shape={tuple(out.last_hidden_state.shape)} "
            f"finite={torch.isfinite(out.last_hidden_state).all().item()} "
            f"|max|={out.last_hidden_state.abs().max().item():.4f} hidden_states={n_hs}"
        )

    try:
        import xformers

        xformers_version = xformers.__version__
    except Exception:
        xformers_version = None

    fixture = {
        "cases": results,
        "meta": {
            "snapshot_revision": SNAPSHOT_REVISION,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "xformers_version": xformers_version,
            "dtype": "float32",
            "has_per_layer_hidden_states": has_hidden_states,
            # The original appends one hidden state PER LAYER (no embedding output); the vendored
            # model emits num_hidden_layers + 1 (embedding first). Align vendored[1:] with reference.
            "hidden_states_layout": "per_layer_only (compare vendored.hidden_states[1:])",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.out)
    print(f"[save] wrote reference fixture to {args.out}")
    print(f"[meta] transformers={transformers.__version__} torch={torch.__version__} xformers={xformers_version}")


if __name__ == "__main__":
    main()
