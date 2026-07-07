"""Validate the pure-torch SwiGLU against ``xformers.ops.SwiGLU`` (F8).

Reference venv only — skipped unless ``xformers`` is importable (it lives only in the pinned
reference venv, never the main env). CPU, fp32: fused GPU kernels add larger diffs, so CPU isolates
the math. GT dims from the NeoBERT checkpoint: in=768, hidden=2048, out=768, bias=False.
"""

import pytest
import torch

xformers_ops = pytest.importorskip("xformers.ops")

from lightning_ir.models.backbones.neobert.modeling_neobert import SwiGLU  # noqa: E402

IN, HIDDEN, OUT = 768, 2048, 768


def test_swiglu_matches_xformers():
    ref = xformers_ops.SwiGLU(IN, HIDDEN, OUT, bias=False)
    mine = SwiGLU(IN, HIDDEN, OUT, bias=False)

    print("xformers keys:", sorted(ref.state_dict()))
    print("vendored keys:", sorted(mine.state_dict()))

    # Load as-is: a name mismatch here means the layout is wrong and must be fixed, not remapped.
    missing, unexpected = mine.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, f"key mismatch missing={missing} unexpected={unexpected}"

    x = torch.randn(4, 16, IN, dtype=torch.float32)
    ref.eval()
    mine.eval()
    with torch.no_grad():
        torch.testing.assert_close(mine(x), ref(x), atol=1e-6, rtol=1e-5)
