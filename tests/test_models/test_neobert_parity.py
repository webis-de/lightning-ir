"""Parity gate: vendored NeoBERT (transformers v5) vs the original remote code (transformers 4.57.6).

Skipped unless BOTH the reference fixture (``scripts/neobert/dump_reference_outputs.py``, run in the
reference venv) and the converted checkpoint exist.

**Gate is frozen** (calibrated from same-architecture measurements — see finding F11 in
``docs/neobert_known_findings.md``). NeoBERT genuinely runs at ``|max|≈130`` (final) / ``~1330``
(intermediate) massive-activation channels, so an *absolute* tolerance is meaningless — the gate is
relative, guarded by a bulk-distribution bound:

- ``finite`` (F6 — the whole point);
- ``cos_min > 0.9999`` (direction);
- magnitude ratio ``|got|_max / |ref|_max ∈ [0.5, 2.0]`` (F9 — no corruption);
- ``max_abs_diff / |ref|_max < 1.5e-2`` (outlier channels; 3.2× over the measured floor);
- ``median_abs_diff < 3e-4`` (bulk — guards O(1) channels the max/cos statistics miss).

**Must run same-architecture as the fixture** (cluster x86): cross-arch fp32 (e.g. Mac ARM) inflates
the residual ~20× on the massive channels. Enforced below when the fixture records its machine.
"""

import platform
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

# Frozen thresholds (F11). Do not loosen without re-measuring the same-arch floor.
MAX_REL_DIFF = 1.5e-2  # max_abs_diff / |ref|_max
MEDIAN_ABS_DIFF = 3e-4
MIN_TOKEN_COS = 0.9999
MAG_RATIO = (0.5, 2.0)

CASES = ["short", "medium", "long", "padded_batch", "pb_row0_single", "pb_row1_single", "pb_row2_single", "pb_row3_single"]


def _masked(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return t[mask.bool()]


@pytest.fixture(scope="module")
def fixture():
    fx = torch.load(FIXTURE, weights_only=False)
    ref_machine = fx.get("meta", {}).get("machine")
    if ref_machine and ref_machine != platform.machine():
        pytest.skip(
            f"parity fixture is arch '{ref_machine}'; running on '{platform.machine()}'. "
            "Cross-arch fp32 inflates the massive-activation residual ~20× (F11) — run same-arch."
        )
    return fx


@pytest.fixture(scope="module")
def model():
    return NeoBERTModel.from_pretrained(CKPT, torch_dtype=torch.float32).eval()


@pytest.mark.parametrize("case", CASES)
def test_parity(case, fixture, model):
    if case not in fixture["cases"]:
        pytest.skip(f"case {case} not in fixture")
    data = fixture["cases"][case]
    m = data["attention_mask"]
    ref = data["last_hidden_state"]
    with torch.no_grad():
        out = model(input_ids=data["input_ids"], attention_mask=m, output_hidden_states=True)
    got = out.last_hidden_state

    assert torch.isfinite(got).all(), f"{case}: non-finite output (F6 regression)"

    a, b = _masked(got, m), _masked(ref, m)
    diff = (a - b).abs()
    ref_mag, got_mag = b.abs().max().item(), a.abs().max().item()
    rel_diff = diff.max().item() / ref_mag
    median_abs = diff.median().item()
    min_cos = F.cosine_similarity(a, b, dim=-1).min().item()

    if data.get("hidden_states") is not None and out.hidden_states is not None:
        got_hs = out.hidden_states[1:]  # n+1 -> drop embedding output to align with reference per-layer
        tbl = [(_masked(g, m) - _masked(r, m)).abs().max().item() for g, r in zip(got_hs, data["hidden_states"])]
        print(f"[{case}] per-layer max_abs_diff L0/mid/last: {tbl[0]:.1e}/{tbl[len(tbl)//2]:.1e}/{tbl[-1]:.1e}")
    print(
        f"[{case}] rel={rel_diff:.2e} median={median_abs:.2e} cos={min_cos:.6f} "
        f"|ref|={ref_mag:.2f} |got|={got_mag:.2f}"
    )

    assert rel_diff < MAX_REL_DIFF, f"{case}: max_abs/|ref|_max {rel_diff:.2e} >= {MAX_REL_DIFF}"
    assert median_abs < MEDIAN_ABS_DIFF, f"{case}: median_abs_diff {median_abs:.2e} >= {MEDIAN_ABS_DIFF}"
    assert min_cos > MIN_TOKEN_COS, f"{case}: min_token_cos {min_cos:.6f} <= {MIN_TOKEN_COS}"
    assert MAG_RATIO[0] * ref_mag <= got_mag <= MAG_RATIO[1] * ref_mag, f"{case}: magnitude {got_mag:.2f} vs {ref_mag:.2f}"


def test_gradient_sanity(fixture, model):
    data = fixture["cases"]["medium"]
    out = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])
    out.last_hidden_state.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    assert any(g.abs().sum() > 0 for g in grads), "all-zero gradients"
