"""lightning-ir integration + regression tests for the vendored NeoBERT backbone (12.4).

Each test encodes a failure that killed a previous NeoBERT attempt (F1-F4). CPU, CI-safe: skipped
unless the converted checkpoint exists.
"""

import subprocess
from pathlib import Path

import pytest
import torch

import lightning_ir  # noqa: F401  triggers _register_backbones()
from lightning_ir import BiEncoderModule
from lightning_ir.models import ColConfig

CKPT = Path("checkpoints/neobert-vendored")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="converted NeoBERT checkpoint not present")


def _col_module():
    return BiEncoderModule(model_name_or_path=str(CKPT), config=ColConfig()).eval()


def test_factory_derivation_finite_forward():
    """The mvrt5 factory builds ColNeoBERTModel over the vendored backbone; a forward is finite."""
    module = _col_module()
    assert type(module.model).__name__ == "ColNeoBERTModel"
    with torch.inference_mode():
        out = module.score("capital of France?", ["Paris is the capital.", "Berlin is in Germany."])
    assert torch.isfinite(out.query_embeddings.embeddings).all()
    assert torch.isfinite(out.doc_embeddings.embeddings).all()
    assert torch.isfinite(out.scores).all()


def test_head_initialization():
    """F3: the lightning-ir-added projection is initialized sanely (finite, non-zero var, no inf)."""
    proj = _col_module().model.projection
    assert torch.isfinite(proj.weight).all()
    assert proj.weight.var().item() > 0
    assert proj.weight.abs().max().item() < 10  # the "inf scores" regression


def test_reload_no_clobber(tmp_path):
    """F2: save -> reload through the lightning-ir path leaves backbone weights untouched."""
    module = _col_module()
    before = {
        k: v.clone()
        for k, v in module.model.state_dict().items()
        if k.startswith("encoder.") or k.startswith("transformer_encoder.0.")
    }
    assert before
    module.model.save_pretrained(tmp_path)
    reloaded = BiEncoderModule(model_name_or_path=str(tmp_path)).model
    after = reloaded.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"backbone tensor clobbered on reload: {k}"


def test_marker_resize_preserves_rows():
    """F1+F4: resize_token_embeddings works (get/set_input_embeddings) and keeps original rows."""
    model = _col_module().model
    emb = model.get_input_embeddings()
    v0 = emb.weight.shape[0]
    orig = emb.weight.data[:v0].clone()
    model.resize_token_embeddings(v0 + 2)
    assert model.get_input_embeddings().weight.shape[0] == v0 + 2
    assert torch.equal(model.get_input_embeddings().weight.data[:v0], orig)


def test_no_factory_edits():
    """The design premise: registration alone suffices — lightning_ir/base/ is untouched vs mvrt5."""
    committed = subprocess.run(
        ["git", "diff", "--name-only", "origin/mvrt5..HEAD", "--", "lightning_ir/base/"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    worktree = subprocess.run(
        ["git", "diff", "--name-only", "--", "lightning_ir/base/"], capture_output=True, text=True
    ).stdout.strip()
    assert committed == "", f"base/ modified vs mvrt5: {committed}"
    assert worktree == "", f"base/ modified in working tree: {worktree}"
