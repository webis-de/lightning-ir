#!/usr/bin/env python3
"""Overfit test: can each head+backbone drive the loss DOWN on a fixed tiny batch?

Motivation: forward-side hypotheses are all exhausted (bf16 faithful, not anisotropic,
ColBERT head geometry healthy — see grad_autopsy / token_geometry / mv_collapse), yet on
the cluster multi-vector (ColBERT/MVR) loss is frozen ~8 while single-vector (DPR) trains.
The canonical "can it learn at all" check: overfit 16 query-doc pairs. If col:neobert
cannot fit what dpr:neobert fits, the symptom is reproduced locally and we instrument it;
if col:neobert fits fine, the frozen cluster loss is config/LR/data, not the model.

Target is a clean, learnable ranking: per query the on-topic doc (index 0 in the fixed
batch) should outscore its 3 off-topic docs by a margin of 4 (same margin-MSE graph as
SupervisedMarginMSE). We sweep a few LRs so a bad LR can't be mistaken for "can't learn",
and split the gradient norm into backbone vs projection to test whether the multi-vector
per-token L2-norm starves the backbone gradient (Jacobian of normalize ~ 1/||proj||).

fp32, forward+backward+AdamW, no cluster, CPU-friendly.

    python scripts/neobert/analysis/mv_overfit.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from grad_autopsy import (  # noqa: E402
    BACKBONES, N_DOCS_PER_QUERY, QUERIES, build, encode_and_score, loss_fn,
)


def grad_norms(model):
    """Total / backbone-only / projection-only gradient L2 norm."""
    tot = bb = proj = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g2 = p.grad.detach().float().pow(2).sum().item()
        tot += g2
        if "projection" in n:
            proj += g2
        else:
            bb += g2
    return tot**0.5, bb**0.5, proj**0.5


def run(combo, lr, steps, device, use_bf16=False, seed=0):
    torch.manual_seed(seed)  # identical projection init across LRs of the same combo
    head, bb = combo.split(":")
    module, model = build(head, BACKBONES[bb])
    model.to(device)

    targets = torch.zeros(len(QUERIES), N_DOCS_PER_QUERY, device=device)
    targets[:, 0] = 4.0  # on-topic doc should lead its 3 negatives by margin 4

    def forward_loss():
        # bf16-mixed exactly like Lightning: autocast the encode/score graph, loss in fp32
        if use_bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                scores = encode_and_score(module, model, device)
            return loss_fn(scores.float(), targets), scores
        scores = encode_and_score(module, model, device)
        return loss_fn(scores, targets), scores

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    traj, gbb0, gproj0, nonfinite = [], None, None, 0
    tag = "bf16" if use_bf16 else "fp32"
    every = max(1, steps // 12)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, scores = forward_loss()
        if not torch.isfinite(scores).all():
            nonfinite += 1
        loss.backward()
        _, gbb, gproj = grad_norms(model)
        if step == 0:
            gbb0, gproj0 = gbb, gproj
        opt.step()
        traj.append(float(loss.detach()))
        if step % every == 0 or step == steps - 1:
            with torch.no_grad():
                sc = scores.detach()
                margin = float(sc[:, 0].mean() - sc[:, 1:].mean())
            print(f"    [{combo} {tag} lr={lr:.0e}] step {step:5d}/{steps}  "
                  f"loss={float(loss.detach()):8.4f}  margin={margin:+.3f}  "
                  f"score_std={float(sc.std()):.4f}  gbb={gbb:.2e}"
                  + (f"  NONFINITE" if nonfinite else ""), flush=True)

    with torch.no_grad():
        _, scores = forward_loss()
        pos = float(scores[:, 0].mean())
        neg = float(scores[:, 1:].mean())
    del model, module
    return dict(combo=combo, lr=lr, traj=traj, gbb0=gbb0, gproj0=gproj0,
                pos=pos, neg=neg, bf16=use_bf16, nonfinite=nonfinite)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combos", nargs="+",
                   default=["col:neobert", "col:bert", "dpr:neobert", "dpr:bert"])
    p.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 1e-3])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--backbone", nargs="*", default=[],
                   help="override backbone paths, e.g. neobert=/path/to/ckpt or neobert=chandar-lab/NeoBERT")
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=["fp32", "bf16", "both"], default="fp32",
                   help="bf16 autocasts the encode/score graph like Lightning's bf16-mixed. "
                        "Use 'both' on the cluster GPU to isolate a precision-only failure.")
    args = p.parse_args()
    for kv in args.backbone:
        k, v = kv.split("=", 1)
        BACKBONES[k] = v
    device = torch.device(args.device)
    precisions = [False, True] if args.precision == "both" else [args.precision == "bf16"]
    print(f"device={device}  steps={args.steps}  lrs={args.lrs}  precision={args.precision}")
    print("target: on-topic doc leads its 3 negatives by margin 4 (margin-MSE).\n")

    best = {}
    for combo in args.combos:
        print(f"===== {combo} =====")
        for use_bf16 in precisions:
            tag = "bf16" if use_bf16 else "fp32"
            for lr in args.lrs:
                r = run(combo, lr, args.steps, device, use_bf16=use_bf16)
                t = r["traj"]
                drop = t[0] - t[-1]
                frac = drop / t[0] if t[0] else 0.0
                marks = [t[0]] + [t[int(args.steps * f) - 1] for f in (0.25, 0.5, 0.75, 1.0)]
                nf = f"  NON-FINITE×{r['nonfinite']}" if r["nonfinite"] else ""
                print(f"  [{tag}] lr={lr:>7.0e}: loss "
                      + " -> ".join(f"{m:6.3f}" for m in marks)
                      + f"   drop={drop:6.3f} ({frac*100:4.1f}%)  "
                      f"final pos/neg={r['pos']:.2f}/{r['neg']:.2f}  "
                      f"g0 bb/proj={r['gbb0']:.2e}/{r['gproj0']:.2e}{nf}")
                key = (combo, tag)
                if key not in best or frac > best[key][1]:
                    best[key] = (lr, frac, r)
        print()

    print("--- verdict (best LR per combo × precision) ---")
    for (combo, tag), (lr, frac, r) in best.items():
        learns = frac > 0.30
        print(f"  {combo:14s} [{tag}]: best {frac*100:5.1f}% loss drop @lr={lr:.0e}  "
              f"-> {'LEARNS' if learns else 'STUCK'}   "
              f"(pos-neg margin={r['pos']-r['neg']:+.2f}, target +4)")


if __name__ == "__main__":
    main()
