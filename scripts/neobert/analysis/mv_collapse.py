#!/usr/bin/env python3
"""Multi-vector collapse probe: does NeoBERT's massive-activation channel collapse the
per-token embedding space *inside the ColBERT head*, starving MaxSim of gradient?

Symptom this explains: ColBERT / MVR (multi-vector) loss frozen (~8, no learning) on
NeoBERT, while DPR (single-vector) trains fine. Multi-vector heads L2-normalize *each
token vector* and score by MaxSim; single-vector pools to one vector. So a backbone
channel that dwarfs all others (token_geometry: max|act|=131.7 vs BERT 9.7, channel 697)
is survivable for DPR but can dominate every projected token vector, making the per-token
normalized vectors collapse toward one axis -> all q_i . d_j ~ const -> MaxSim flat ->
no gradient.

The raw-space token_geometry probe could NOT see this: it measured geometry *before* the
projection + per-token L2-norm. This probe measures it *after* the head, and adds a causal
control: zero the massive channel before the projection and see if the collapse lifts.

Metric: per document, take the masked, post-head, L2-normalized token matrix E (n_tok x D)
and report the top-1 singular-value energy share  s1^2 / sum(si^2).  1.0 = every token
vector points the same way (fully collapsed, MaxSim degenerate); ~1/rank = healthy spread.

Forward passes only, no grad, no training, CPU-friendly. Reuses grad_autopsy's fixed batch
and model builder so the two probes stay consistent.

Usage:
    python scripts/neobert/analysis/mv_collapse.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from grad_autopsy import BACKBONES, DOCS, QUERIES, build  # noqa: E402


def head_embeddings(model, enc, input_type, ablate_channel=None):
    """Replicate ColModel.encode's head math so we can optionally ablate a backbone channel
    *before* the projection. Returns (raw_hidden, post_head_normalized, scoring_mask)."""
    h = model._backbone_forward(**enc).last_hidden_state  # (B, S, H)
    h_head = h
    if ablate_channel is not None:
        h_head = h.clone()
        h_head[..., ablate_channel] = 0.0
    proj = model.projection(h_head)
    if getattr(model.config, "normalization_strategy", None) == "l2":
        emb = F.normalize(proj, dim=-1)
    else:
        emb = proj
    mask = model.scoring_mask(enc, input_type)
    return h, emb, mask.bool()


def top1_share(vectors: torch.Tensor) -> float:
    """Energy fraction in the leading singular direction of a (n, d) matrix. 1.0 => rank-1
    collapse (all rows colinear); ~1/min(n,d) => isotropic."""
    if vectors.shape[0] < 2:
        return float("nan")
    sv = torch.linalg.svdvals(vectors.float())
    return float((sv[0] ** 2 / (sv**2).sum()).item())


def per_doc_collapse(emb: torch.Tensor, mask: torch.Tensor):
    """Mean over docs of top-1 SV share and mean|pairwise-cosine| of the masked token vectors."""
    shares, abscos = [], []
    for b in range(emb.shape[0]):
        vecs = emb[b][mask[b]]  # (n_tok, D), already unit-norm post L2
        if vecs.shape[0] < 2:
            continue
        shares.append(top1_share(vecs))
        gram = vecs @ vecs.T
        off = gram[~torch.eye(gram.shape[0], dtype=torch.bool)]
        abscos.append(float(off.abs().mean().item()))
    n = max(len(shares), 1)
    return sum(shares) / n, sum(abscos) / n


def maxsim_saturation(q_emb, q_mask, d_emb, d_mask):
    """For query 0 vs its 4 docs: std of the q_i.d_j similarity entries (low std = saturated,
    every token pair scores alike so MaxSim carries ~no gradient) and the MaxSim scores."""
    qv = q_emb[0][q_mask[0]]  # (nq, D)
    stds, maxsims = [], []
    for b in range(d_emb.shape[0]):
        dv = d_emb[b][d_mask[b]]  # (nd, D)
        if qv.shape[0] < 1 or dv.shape[0] < 1:
            continue
        sims = qv @ dv.T  # (nq, nd) in [-1, 1]
        stds.append(float(sims.std().item()))
        maxsims.append(float(sims.max(dim=1).values.sum().item()))
    return stds, maxsims


def analyze(label, backbone_path, device):
    module, model = build("col", backbone_path)
    model.to(device).eval()
    tok = module.tokenizer
    enc = tok.tokenize(QUERIES, DOCS, return_tensors="pt", padding=True, truncation=True)
    q_enc = {k: v.to(device) for k, v in enc["query_encoding"].items()}
    d_enc = {k: v.to(device) for k, v in enc["doc_encoding"].items()}

    with torch.no_grad():
        h_d, e_d, m_d = head_embeddings(model, d_enc, "doc")
        h_q, e_q, m_q = head_embeddings(model, q_enc, "query")

        # locate the dominant backbone channel on the used doc tokens
        used = h_d[m_d]  # (sum_tok, H)
        chan_mag = used.abs().mean(dim=0)  # mean |act| per channel over real tokens
        c = int(chan_mag.argmax().item())
        cmax = float(used.abs().amax().item())

        # pre-head collapse (raw hidden, unit-normalized so it is comparable)
        pre_share, pre_abscos = per_doc_collapse(F.normalize(h_d, dim=-1), m_d)
        # post-head collapse (what MaxSim actually sees)
        post_share, post_abscos = per_doc_collapse(e_d, m_d)
        # causal control: kill channel c before the projection, re-measure post-head
        _, e_d_ab, _ = head_embeddings(model, d_enc, "doc", ablate_channel=c)
        abl_share, abl_abscos = per_doc_collapse(e_d_ab, m_d)

        stds, _ = maxsim_saturation(e_q, m_q, e_d, m_d)

    print(f"\n== col:{label}  ({backbone_path}) ==")
    print(f"  dominant channel idx={c}  mean|act|={chan_mag[c]:.2f}  max|act|={cmax:.1f}  "
          f"(median chan mean|act|={chan_mag.median():.3f})")
    print(f"  top-1 SV share  pre-head={pre_share:.3f}  post-head={post_share:.3f}  "
          f"post-head[ch{c}->0]={abl_share:.3f}")
    print(f"  mean|cos|       pre-head={pre_abscos:.3f}  post-head={post_abscos:.3f}  "
          f"post-head[ch{c}->0]={abl_abscos:.3f}")
    print(f"  MaxSim sim-matrix std over q0's 4 docs = "
          f"[{', '.join(f'{s:.3f}' for s in stds)}]")
    return dict(label=label, chan=c, post_share=post_share, abl_share=abl_share,
                pre_share=pre_share, sim_std=stds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["neobert", "bert"],
                   help="backbone keys from grad_autopsy.BACKBONES")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    print(f"device={device}")
    print("top-1 SV share: 1.0 = all token vectors colinear (MaxSim degenerate); "
          "lower = healthier spread.")
    rows = [analyze(m, BACKBONES[m], device) for m in args.models]

    print("\n--- verdict ---")
    for r in rows:
        collapsed = r["post_share"] > 0.6
        lifts = (r["post_share"] - r["abl_share"]) > 0.15
        print(f"  col:{r['label']}: post-head collapse={'YES' if collapsed else 'no'} "
              f"({r['post_share']:.2f}); ablating ch{r['chan']} lifts it="
              f"{'YES' if lifts else 'no'} (-> {r['abl_share']:.2f})")


if __name__ == "__main__":
    main()
