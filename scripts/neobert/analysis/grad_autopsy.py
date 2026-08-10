#!/usr/bin/env python3
"""Gradient autopsy: is bf16-mixed destroying the training signal for multi-vector heads
on NeoBERT?

For each head×backbone combo (default: col:neobert, col:bert, dpr:neobert, dpr:bert) this
builds the lightning-ir model exactly like training does (same ColConfig as the failed run,
add_marker_tokens=false), runs the SAME batch forward+backward twice — once in fp32, once
under torch.autocast(bfloat16), i.e. Lightning's bf16-mixed — and compares the gradients:

  * cosine(grad_bf16, grad_fp32) per parameter group  (1.0 = faithful, ~0 = noise)
  * norm ratio ||g16|| / ||g32|| per group
  * forward fidelity: max |score_bf16 - score_fp32|, loss values

H2 signature: col:neobert shows low gradient cosine (projection / late layers worst) while
col:bert and dpr:neobert stay near 1.0.

Forward+backward only, no optimizer steps, no files touched outside --outdir.
Self-contained: ships a built-in mini-batch, so no dataset access is needed.

Usage (repo root, branch venv):
    python scripts/neobert/analysis/grad_autopsy.py --outdir docs/figures/grad_autopsy
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

try:
    import lightning_ir  # noqa: F401  (registers the vendored neobert)
    from lightning_ir import BiEncoderModule
    from lightning_ir.bi_encoder.bi_encoder_model import BiEncoderOutput
except Exception as e:  # pragma: no cover
    sys.exit(f"needs the lightning-ir branch venv: `import lightning_ir` failed: {e}")

try:
    from lightning_ir.models import ColConfig, DprConfig
except ImportError:
    from lightning_ir.models.bi_encoders.col import ColConfig
    from lightning_ir.models.bi_encoders.dpr import DprConfig

torch.manual_seed(0)

BACKBONES = {
    "neobert": "checkpoints/neobert-vendored",
    "bert": "bert-base-uncased",
    "modernbert": "answerdotai/ModernBERT-base",
}

# same ColConfig as the failed run (jolly-pyramid-8 YAML), minus dataset-side options
COL_KWARGS = dict(
    query_length=32, doc_length=256, similarity_function="dot",
    normalization_strategy="l2", add_marker_tokens=False,
    query_mask_scoring_tokens=None, doc_mask_scoring_tokens="punctuation",
    query_aggregation_function="mean", doc_aggregation_function="max",
    embedding_dim=128, projection="linear",
    query_expansion=False, doc_expansion=False,
)

QUERIES = [
    "what causes northern lights",
    "how long do red blood cells live",
    "difference between alligator and crocodile",
    "when was the eiffel tower built",
]
DOCS = [  # 4 docs per query: 1 on-topic + 3 off-topic (order fixed, targets synthetic anyway)
    "The aurora borealis appears when charged particles from the sun collide with gases in "
    "Earth's upper atmosphere, producing shimmering curtains of green and red light near the poles.",
    "Basalt is a fine-grained volcanic rock formed from rapidly cooled lava flows.",
    "The recipe calls for two cups of flour, a pinch of salt, and slow kneading by hand.",
    "Municipal bonds are debt securities issued by states, cities, and counties.",
    "Human red blood cells survive for roughly 120 days before the spleen removes them from "
    "circulation; the bone marrow continuously produces replacements.",
    "The steam locomotive transformed freight transport during the nineteenth century.",
    "Photosynthesis converts carbon dioxide and water into glucose using light energy.",
    "A capacitor stores electrical energy in an electric field between two conductors.",
    "Alligators have broad U-shaped snouts and tolerate only freshwater, while crocodiles show "
    "narrow V-shaped snouts, visible lower teeth, and thrive in salt water as well.",
    "The novel opens with a long description of provincial life in nineteenth-century Russia.",
    "Regular aerobic exercise strengthens the heart muscle and lowers resting blood pressure.",
    "The court ruled that the statute exceeded the agency's delegated authority.",
    "Construction of the Eiffel Tower finished in 1889 for the Paris World's Fair; Gustave "
    "Eiffel's company assembled its eighteen thousand iron parts in just over two years.",
    "Quantum entanglement links particle states regardless of the distance between them.",
    "Sourdough fermentation depends on wild yeast and lactic acid bacteria in the starter.",
    "The glacier retreated four hundred meters over the last two decades of observation.",
]
N_DOCS_PER_QUERY = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--combos", nargs="+",
                   default=["col:neobert", "col:bert", "dpr:neobert", "dpr:bert"],
                   help="head:backbone pairs; heads: col|dpr, backbones: see BACKBONES")
    p.add_argument("--backbone", nargs="*", default=[],
                   help="override backbone paths, label=path")
    p.add_argument("--n-batches", type=int, default=3,
                   help="repeat with shuffled doc order, report means")
    p.add_argument("--device", default=None, help="cpu|cuda (default: cuda if available else cpu)")
    p.add_argument("--outdir", type=Path, default=Path("docs/figures/grad_autopsy"))
    return p.parse_args()


def build(head: str, backbone_path: str):
    cfg = ColConfig(**COL_KWARGS) if head == "col" else DprConfig()
    module = BiEncoderModule(model_name_or_path=backbone_path, config=cfg)
    model = module.model
    model.train(False)  # kill dropout for a deterministic bf16-vs-fp32 comparison
    for p_ in model.parameters():
        p_.requires_grad_(True)
    return module, model


def encode_and_score(module, model, device):
    """Tokenize the fixed batch and return scores (n_q, n_docs) through the model's own
    encode/score path (the same graph training uses)."""
    tok = module.tokenizer
    if hasattr(tok, "tokenize"):
        enc = tok.tokenize(QUERIES, DOCS, return_tensors="pt", padding=True, truncation=True)
        q_enc, d_enc = enc["query_encoding"], enc["doc_encoding"]
    else:  # pragma: no cover — fallback if the tokenizer API differs on this branch
        sys.exit(f"unexpected tokenizer API on this branch: {type(tok)} has no .tokenize; "
                 "adapt encode_and_score() to the branch's LightningIRTokenizer.")
    q_enc = {k: v.to(device) for k, v in q_enc.items()}
    d_enc = {k: v.to(device) for k, v in d_enc.items()}
    for name in ("encode_query", "encode_doc", "score"):
        if not hasattr(model, name):
            sys.exit(f"model {type(model).__name__} lacks .{name}; available: "
                     f"{[m for m in dir(model) if 'encod' in m or 'score' in m]} — adapt here.")
    q = model.encode_query(q_enc)
    d = model.encode_doc(d_enc)
    num_docs = [N_DOCS_PER_QUERY] * len(QUERIES)
    output = BiEncoderOutput(query_embeddings=q, doc_embeddings=d)
    output = model.score(output, num_docs)
    return output.scores.view(len(QUERIES), N_DOCS_PER_QUERY)


def loss_fn(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Margin-MSE against fixed synthetic teacher margins — same graph shape as
    SupervisedMarginMSE; supervision content is irrelevant for gradient *fidelity*."""
    s_margin = scores[:, :1] - scores[:, 1:]
    t_margin = targets[:, :1] - targets[:, 1:]
    return torch.nn.functional.mse_loss(s_margin, t_margin)


def grads_for(module, model, device, use_bf16: bool, targets):
    model.zero_grad(set_to_none=True)
    if use_bf16:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            scores = encode_and_score(module, model, device)
            loss = loss_fn(scores.float(), targets)
    else:
        scores = encode_and_score(module, model, device)
        loss = loss_fn(scores, targets)
    loss.backward()
    grads = {n: p.grad.detach().float().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return grads, scores.detach().float(), float(loss)


def make_grouper(model):
    n_layers = getattr(model.config, "num_hidden_layers", 12)

    def group(name: str) -> str:
        if "projection" in name:
            return "projection"
        m = re.search(r"\.(\d+)\.", name)
        if m:
            i = int(m.group(1))
            return ("layers_early" if i < n_layers / 3
                    else "layers_mid" if i < 2 * n_layers / 3 else "layers_late")
        if "embed" in name or name.endswith("encoder.weight"):
            return "embeddings"
        return "other"
    return group


def compare(g16: dict, g32: dict, group) -> dict:
    buckets16, buckets32 = defaultdict(list), defaultdict(list)
    for n in g32:
        if n in g16:
            buckets16[group(n)].append(g16[n].flatten())
            buckets32[group(n)].append(g32[n].flatten())
    out = {}
    for grp in sorted(buckets32):
        a = torch.cat(buckets16[grp]); b = torch.cat(buckets32[grp])
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        ratio = (a.norm() / (b.norm() + 1e-12)).item()
        out[grp] = (cos, ratio)
    a = torch.cat([g16[n].flatten() for n in g32 if n in g16])
    b = torch.cat([g32[n].flatten() for n in g32 if n in g16])
    out["ALL"] = (torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                  (a.norm() / (b.norm() + 1e-12)).item())
    return out


def main():
    global DOCS
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for kv in args.backbone:
        k, v = kv.split("=", 1)
        BACKBONES[k] = v
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "mps":
        print("NOTE: autocast-bf16 on MPS differs from CUDA/CPU; using CPU is the safer proxy.")
    print(f"device={device}")

    rows = []
    for combo in args.combos:
        head, bb = combo.split(":")
        print(f"\n== {combo}  ({BACKBONES[bb]}) ==")
        module, model = build(head, BACKBONES[bb])
        model.to(device)
        group = make_grouper(model)
        targets = torch.randn(len(QUERIES), N_DOCS_PER_QUERY, generator=torch.Generator().manual_seed(1))
        agg = defaultdict(lambda: [0.0, 0.0])
        fwd_diffs, l32s, l16s = [], [], []
        base_docs = list(DOCS)
        for b in range(args.n_batches):
            rng = torch.Generator().manual_seed(100 + b)
            perm = torch.randperm(len(base_docs) // N_DOCS_PER_QUERY, generator=rng)
            # shuffle query blocks (keeps 4-docs-per-query structure, varies the batch)
            DOCS = [base_docs[q * N_DOCS_PER_QUERY + i] for q in perm.tolist()
                    for i in range(N_DOCS_PER_QUERY)]
            g32, s32, l32 = grads_for(module, model, device, use_bf16=False, targets=targets)
            g16, s16, l16 = grads_for(module, model, device, use_bf16=True, targets=targets)
            fwd_diffs.append((s16 - s32).abs().max().item())
            l32s.append(l32); l16s.append(l16)
            for grp, (cos, ratio) in compare(g16, g32, group).items():
                agg[grp][0] += cos / args.n_batches
                agg[grp][1] += ratio / args.n_batches
        DOCS = base_docs
        print(f"  max|score_bf16-score_fp32| = {max(fwd_diffs):.4g}   "
              f"loss fp32/bf16 = {sum(l32s)/len(l32s):.4f}/{sum(l16s)/len(l16s):.4f}")
        for grp, (cos, ratio) in sorted(agg.items()):
            print(f"  {grp:14s} cos(g16,g32)={cos:7.4f}   |g16|/|g32|={ratio:6.3f}")
            rows.append((combo, grp, cos, ratio, max(fwd_diffs)))
        del model, module

    lines = ["| combo | group | cos(g16,g32) | norm ratio | max fwd score diff |",
             "|---|---|---|---|---|"]
    for combo, grp, cos, ratio, fd in rows:
        lines.append(f"| {combo} | {grp} | {cos:.4f} | {ratio:.3f} | {fd:.4g} |")
    (args.outdir / "grad_autopsy.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        combos = sorted({r[0] for r in rows})
        groups = [g for g in ["embeddings", "layers_early", "layers_mid",
                              "layers_late", "projection", "ALL"]
                  if any(r[1] == g for r in rows)]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        w = 0.8 / len(combos)
        for i, c in enumerate(combos):
            vals = [next((r[2] for r in rows if r[0] == c and r[1] == g), float("nan"))
                    for g in groups]
            ax.bar([x + i * w for x in range(len(groups))], vals, w, label=c)
        ax.set_xticks([x + w * (len(combos) - 1) / 2 for x in range(len(groups))])
        ax.set_xticklabels(groups); ax.set_ylabel("cos(grad_bf16, grad_fp32)")
        ax.axhline(1.0, color="gray", lw=0.6); ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        ax.set_title("gradient fidelity under bf16-mixed, by parameter group")
        fig.tight_layout(); fig.savefig(args.outdir / "grad_autopsy.png", dpi=150)
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")

    print(f"\nwritten: {args.outdir}/grad_autopsy.md (+ .png)")


if __name__ == "__main__":
    main()