import torch
from typing import Tuple


def compute_cos_sin(
    seq_len: int,
    dim: int,
    theta: float,
    device: torch.device,
    position_ids: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real-valued RoPE cos/sin, computed on the fly in fp32.

    Numerically identical to the original complex rotary implementation (same ``inv_freq``, same
    rotation), but with **no precomputed complex64 buffer** — so transformers-v5's meta-device
    loading and dtype casting can't leave it uninitialized/corrupted.

    Returns cos, sin of shape (seq_len, dim // 2), fp32.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: (dim // 2)] / dim))
    if position_ids is not None:
        # Only 1-D (packed/flat) position_ids are supported; 2-D per-example ids would need
        # position-indexed rotary tables (the original) and don't broadcast here.
        assert position_ids.dim() == 1, "compute_cos_sin supports only 1-D position_ids (got %dD)." % position_ids.dim()
        t = position_ids.to(device=device, dtype=torch.float32)
    else:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim // 2)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings with real cos/sin, matching the original interleaved
    (consecutive-pair) complex formulation exactly.

    The original does ``view_as_complex(x.reshape(..., -1, 2))``, which pairs *consecutive*
    elements: complex[i] = x[2i] + i*x[2i+1]. Multiplying by (cos + i*sin) gives
        out[2i]   = x[2i]*cos - x[2i+1]*sin
        out[2i+1] = x[2i]*sin + x[2i+1]*cos
    which is reproduced here in real arithmetic.

    xq, xk: (batch, seq, heads, dim_head). cos, sin: (seq, dim_head // 2).
    """
    cos = cos[None, :, None, :]  # (1, seq, 1, d/2)
    sin = sin[None, :, None, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        x1 = xf[..., 0::2]  # even indices  -> real part
        x2 = xf[..., 1::2]  # odd indices   -> imag part
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        out = torch.stack((o1, o2), dim=-1).flatten(-2)  # re-interleave to (..., dim_head)
        return out.type_as(x)

    return rotate(xq), rotate(xk)
