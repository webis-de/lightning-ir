"""Parity gate: vendored NeoBERT (transformers v5) vs the original remote code (transformers 4.57.6).

Skipped unless BOTH the reference fixture (produced by ``scripts/neobert/dump_reference_outputs.py``
in the reference venv) and the converted checkpoint exist. Tolerances are fixed by the spec (11.3):
finite; ``max_abs_diff < 2e-3`` (expected 1e-4..1e-3); ``min_token_cos > 0.9999``; sane magnitude.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from lightning_ir.models.backbones.neobert import NeoBERTModel

CKPT = Path("checkpoints/neobert-vendored")
FIXTURE = Path("tests/test_models/fixtures/reference_outputs.pt")

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and FIXTURE.exists()),
    reason="need converted checkpoint + reference fixture",
)

MAX_ABS_DIFF = 2e-3
MIN_TOKEN_COS = 0.9999


def _masked(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return t[mask.bool()]


@pytest.fixture(scope="module")
def fixture():
    return torch.load(FIXTURE, weights_only=False)


@pytest.fixture(scope="module")
def model():
    return NeoBERTModel.from_pretrained(CKPT, torch_dtype=torch.float32).eval()


@pytest.mark.parametrize("case", ["short", "medium", "long", "padded_batch"])
def test_parity(case, fixture, model):
    data = fixture["cases"][case]
    input_ids, attention_mask = data["input_ids"], data["attention_mask"]
    ref = data["last_hidden_state"]
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    got = out.last_hidden_state

    assert torch.isfinite(got).all(), f"{case}: non-finite output (F6 regression)"

    m = attention_mask
    max_abs_diff = (_masked(got, m) - _masked(ref, m)).abs().max().item()
    min_cos = F.cosine_similarity(_masked(got, m), _masked(ref, m), dim=-1).min().item()
    ref_mag, got_mag = _masked(ref, m).abs().max().item(), _masked(got, m).abs().max().item()

    # per-layer table (vendored emits n+1; align vendored[1:] with reference per-layer states)
    if data.get("hidden_states") is not None and out.hidden_states is not None:
        ref_hs = data["hidden_states"]
        got_hs = out.hidden_states[1:]  # drop embedding output
        table = [
            (i, (_masked(g, m) - _masked(r, m)).abs().max().item()) for i, (g, r) in enumerate(zip(got_hs, ref_hs))
        ]
        probe = [table[0], table[len(table) // 2], table[-1]]
        print(f"[{case}] per-layer max_abs_diff (layer0, mid, last): {probe}")

    print(f"[{case}] max_abs_diff={max_abs_diff:.2e} min_cos={min_cos:.6f} |ref|={ref_mag:.3f} |got|={got_mag:.3f}")

    assert max_abs_diff < MAX_ABS_DIFF, f"{case}: max_abs_diff {max_abs_diff:.2e} >= {MAX_ABS_DIFF}"
    assert min_cos > MIN_TOKEN_COS, f"{case}: min_token_cos {min_cos:.6f} <= {MIN_TOKEN_COS}"
    assert 0.5 * ref_mag <= got_mag <= 2.0 * ref_mag, f"{case}: magnitude {got_mag:.3f} vs ref {ref_mag:.3f}"


def test_gradient_sanity(fixture, model):
    data = fixture["cases"]["medium"]
    out = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])
    loss = out.last_hidden_state.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    assert any(g.abs().sum() > 0 for g in grads), "all-zero gradients"
