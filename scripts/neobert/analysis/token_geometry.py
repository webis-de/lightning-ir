#!/usr/bin/env python3
"""Token-geometry analysis: is NeoBERT's token space too anisotropic for multi-vector IR?

Compares backbone last-hidden-state token embeddings across models (H1 test from the
multi-vector diagnosis): cosine-similarity distributions, rogue-channel profile,
channel-ablation effect, effective rank.

Forward passes only, fp32, CPU or GPU. No training, no writes outside --outdir.

Usage (from the lightning-ir repo root, venv with the neobert-backbone branch):

    python scripts/neobert/analysis/token_geometry.py \
        --passages passages.txt \
        --models bert=bert-base-uncased modernbert=answerdotai/ModernBERT-base \
                 neobert=checkpoints/neobert-vendored \
        --outdir docs/figures/token_geometry

passages.txt: one passage per line, ~200 lines (MS MARCO sample). If --passages is
omitted, tries ir_datasets msmarco-passage (first --n-passages docs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:  # registers the vendored neobert with the Auto classes
    import lightning_ir  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"WARNING: `import lightning_ir` failed ({e}); hub models still work, "
          "the vendored neobert path will not.", file=sys.stderr)

from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)
np.random.seed(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--passages", type=Path, default=None)
    p.add_argument("--n-passages", type=int, default=200)
    p.add_argument("--models", nargs="+",
                   default=["bert=bert-base-uncased",
                            "modernbert=answerdotai/ModernBERT-base",
                            "neobert=checkpoints/neobert-vendored"],
                   help="label=path_or_hub_id pairs")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ablate-topk", nargs="+", type=int, default=[1, 3, 8])
    p.add_argument("--n-pairs", type=int, default=100_000,
                   help="sampled cross-doc token pairs for the anisotropy statistic")
    p.add_argument("--outdir", type=Path, default=Path("docs/figures/token_geometry"))
    return p.parse_args()


def fetch_msmarco_http(n: int) -> list[str]:
    """Pull MS MARCO passages via the HF datasets-server REST API (a few hundred KB,
    no extra libraries, no 3 GB collection download). Spread over several offsets."""
    import json
    import urllib.request

    out: list[str] = []
    for offset in (0, 20_000, 40_000, 60_000, 80_000):
        url = ("https://datasets-server.huggingface.co/rows"
               "?dataset=microsoft%2Fms_marco&config=v1.1&split=train"
               f"&offset={offset}&length=100")
        with urllib.request.urlopen(url, timeout=60) as r:
            rows = json.load(r)["rows"]
        for row in rows:
            for txt in row["row"]["passages"]["passage_text"]:
                t = " ".join(txt.split())
                if len(t) > 100:
                    out.append(t)
        if len(out) >= 3 * n:
            break
    seen, deduped = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped[:n]


def load_passages(args) -> list[str]:
    """Priority: --passages file  >  local ir_datasets cache  >  HTTP fetch (HF API)."""
    if args.passages is not None:
        lines = [l.strip() for l in args.passages.read_text().splitlines() if l.strip()]
        if not lines:
            sys.exit(f"no passages in {args.passages}")
        return lines[: args.n_passages]
    try:
        import ir_datasets
        ds = ir_datasets.load("msmarco-passage")
        out = []
        for doc in ds.docs_iter():
            out.append(doc.text)
            if len(out) >= args.n_passages:
                break
        print("passages: ir_datasets (local cache)")
        return out
    except Exception:
        pass
    try:
        out = fetch_msmarco_http(args.n_passages)
        if len(out) < args.n_passages // 2:
            raise RuntimeError(f"only {len(out)} passages fetched")
        print(f"passages: fetched {len(out)} via HF datasets-server API")
        (args.outdir / "passages.txt").parent.mkdir(parents=True, exist_ok=True)
        (args.outdir / "passages.txt").write_text("\n".join(out) + "\n")
        print(f"  saved copy to {args.outdir / 'passages.txt'} (reuse with --passages for reproducibility)")
        return out
    except Exception as e:
        sys.exit(f"no --passages, no ir_datasets, and HTTP fetch failed: {e}\n"
                 "Fix: create passages.txt manually (one passage per line) and pass --passages.")


@torch.no_grad()
def embed(model_path: str, passages: list[str], max_length: int, batch_size: int,
          device: torch.device):
    """Return (tokens[N, H] fp32 on CPU, doc_ids[N]) for all non-special, non-pad tokens."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
    print(f"  loaded {model_path} -> {type(model).__name__} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    all_vecs, all_docs = [], []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt", return_special_tokens_mask=True)
        special = enc.pop("special_tokens_mask").bool()
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**{k: v for k, v in enc.items() if k in
                       ("input_ids", "attention_mask", "token_type_ids")})
        h = out.last_hidden_state.float().cpu()                     # (B, L, H)
        keep = enc["attention_mask"].bool().cpu() & ~special        # real, non-special tokens
        for b in range(h.shape[0]):
            v = h[b][keep[b]]
            all_vecs.append(v)
            all_docs.append(torch.full((v.shape[0],), i + b, dtype=torch.long))
    del model
    return torch.cat(all_vecs), torch.cat(all_docs)


def cos_stats(vecs: torch.Tensor, docs: torch.Tensor, n_pairs: int, tag: str):
    """Within-doc pairwise cosines (all pairs, capped per doc) + sampled cross-doc cosines."""
    v = torch.nn.functional.normalize(vecs, dim=-1)
    within = []
    for d in torch.unique(docs):
        m = v[docs == d]
        if m.shape[0] < 2:
            continue
        c = (m @ m.T)
        iu = torch.triu_indices(c.shape[0], c.shape[0], offset=1)
        within.append(c[iu[0], iu[1]])
    within = torch.cat(within)
    idx_a = torch.randint(0, v.shape[0], (n_pairs,))
    idx_b = torch.randint(0, v.shape[0], (n_pairs,))
    diff_doc = docs[idx_a] != docs[idx_b]
    cross = (v[idx_a[diff_doc]] * v[idx_b[diff_doc]]).sum(-1)
    def s(x): return dict(mean=x.mean().item(), median=x.median().item(),
                          p5=x.quantile(0.05).item(), p95=x.quantile(0.95).item())
    return {"tag": tag, "within": s(within), "cross": s(cross),
            "_within_raw": within, "_cross_raw": cross}


def effective_rank(vecs: torch.Tensor, max_tokens: int = 20_000):
    x = vecs[torch.randperm(vecs.shape[0])[:max_tokens]]
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = (s ** 2) / (s ** 2).sum()
    erank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    return erank, p[:5].tolist()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon; fp32 inference, fine for statistics
    else:
        device = torch.device("cpu")
    passages = load_passages(args)
    print(f"{len(passages)} passages, device={device}")

    models = dict(kv.split("=", 1) for kv in args.models)
    results, raw_for_plots = {}, {}

    for label, path in models.items():
        print(f"\n== {label} ==")
        vecs, docs = embed(path, passages, args.max_length, args.batch_size, device)
        print(f"  {vecs.shape[0]} tokens, hidden={vecs.shape[1]}, "
              f"|max| activation = {vecs.abs().max().item():.1f}")

        chan = vecs.abs().mean(0)                                   # per-channel mean |act|
        top = torch.argsort(chan, descending=True)
        top10 = [(int(i), float(chan[i])) for i in top[:10]]

        base = cos_stats(vecs, docs, args.n_pairs, "full")
        ablations = {}
        for k in args.ablate_topk:
            v2 = vecs.clone()
            v2[:, top[:k]] = 0.0
            ablations[k] = cos_stats(v2, docs, args.n_pairs, f"ablate-top{k}")
        erank, ev5 = effective_rank(vecs)

        results[label] = dict(n_tokens=int(vecs.shape[0]),
                              max_abs=float(vecs.abs().max()),
                              top10_channels=top10,
                              cos_full=base, cos_ablate=ablations,
                              erank=erank, top5_evr=ev5)
        raw_for_plots[label] = dict(base=base, ablations=ablations,
                                    chan_sorted=chan.sort(descending=True).values.numpy())
        print(f"  cross-doc mean cos = {base['cross']['mean']:.3f}  "
              f"(ablate-top3: {ablations.get(3, base)['cross']['mean']:.3f})   "
              f"erank = {erank:.1f}")

    # ---------- figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"bert": "#4878CF", "modernbert": "#6ACC65", "neobert": "#D65F5F"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for label in models:
        c = colors.get(label, None)
        r = raw_for_plots[label]
        ax[0, 0].hist(r["base"]["_within_raw"].numpy(), bins=200, density=True,
                      histtype="step", lw=1.8, label=label, color=c)
        ax[0, 1].hist(r["base"]["_cross_raw"].numpy(), bins=200, density=True,
                      histtype="step", lw=1.8, label=label, color=c)
        ks = sorted(r["ablations"])
        ax[1, 0].plot([0] + ks,
                      [r["base"]["cross"]["mean"]] +
                      [r["ablations"][k]["cross"]["mean"] for k in ks],
                      marker="o", label=label, color=c)
        ax[1, 1].semilogy(r["chan_sorted"][:64], marker=".", ms=3, label=label, color=c)
    ax[0, 0].set_title("within-passage token cosine"); ax[0, 0].set_xlim(-1, 1)
    ax[0, 1].set_title("cross-passage token cosine (anisotropy)"); ax[0, 1].set_xlim(-1, 1)
    ax[1, 0].set_title("mean cross-doc cosine vs #ablated top channels")
    ax[1, 0].set_xlabel("channels zeroed"); ax[1, 0].axhline(0, lw=0.5, color="gray")
    ax[1, 1].set_title("per-channel mean |activation| (sorted, top 64)")
    for a in ax.flat:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.outdir / "token_geometry.png", dpi=150)

    # ---------- summary table ----------
    lines = ["| model | tokens | max\\|act\\| | erank | cross-cos mean | ablate-top3 | top channel (idx:val) |",
             "|---|---|---|---|---|---|---|"]
    for label, r in results.items():
        t0 = r["top10_channels"][0]
        lines.append(f"| {label} | {r['n_tokens']} | {r['max_abs']:.1f} | {r['erank']:.1f} "
                     f"| {r['cos_full']['cross']['mean']:.3f} "
                     f"| {r['cos_ablate'].get(3, r['cos_full'])['cross']['mean']:.3f} "
                     f"| {t0[0]}:{t0[1]:.1f} |")
    table = "\n".join(lines)
    (args.outdir / "summary.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nfigures + summary written to {args.outdir}/")


if __name__ == "__main__":
    main()