"""Standalone SwiGLU equivalence check (F8): pure-torch SwiGLU vs xformers.ops.SwiGLU.

Run in venv_neobert_ref (has xformers), in any dir. Pure torch + xformers only — no transformers,
no lightning_ir. Confirms whether the ~5e-3-relative same-arch parity residual is the FFN kernel.

    source venv_neobert_ref/bin/activate
    python swiglu_equiv_check.py

NeoBERT dims: in=768, hidden=2048, out=768, bias=False.
"""

import torch
import torch.nn.functional as F
from torch import nn

IN, HIDDEN, OUT = 768, 2048, 768


class SwiGLU(nn.Module):
    """The vendored pure-torch SwiGLU (fused w12 / w3), identical to modeling_neobert.SwiGLU."""

    def __init__(self, in_features, hidden_features, out_features, bias=False):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def main():
    import xformers.ops

    ref = xformers.ops.SwiGLU(IN, HIDDEN, OUT, bias=False)
    mine = SwiGLU(IN, HIDDEN, OUT, bias=False)
    print("xformers keys:", sorted(ref.state_dict()))
    print("vendored keys:", sorted(mine.state_dict()))

    missing, unexpected = mine.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, f"key mismatch missing={missing} unexpected={unexpected}"

    ref.eval()
    mine.eval()
    # small inputs and LARGE inputs (the parity residual lives on ~1330-magnitude activations)
    for scale in (1.0, 50.0):
        x = torch.randn(4, 64, IN, dtype=torch.float32) * scale
        with torch.no_grad():
            a, b = mine(x), ref(x)
        d = (a - b).abs()
        big = b.abs() > 1.0
        rel = (d[big] / b.abs()[big]).max().item() if big.any() else 0.0
        print(f"scale={scale:>5}: max_abs_diff={d.max().item():.3e}  max_rel(|out|>1)={rel:.3e}  |out|max={b.abs().max().item():.1f}")


if __name__ == "__main__":
    main()
