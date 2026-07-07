"""Unit tests for the vendored NeoBERT backbone (CPU, CI-safe).

Skipped unless the converted checkpoint ``checkpoints/neobert-vendored`` exists (it is gitignored
and produced by ``scripts/neobert/convert_checkpoint.py``), so CI stays green without it.
"""

from pathlib import Path

import pytest
import torch

from lightning_ir.models.backbones.neobert import NeoBERTForMaskedLM

CKPT = Path("checkpoints/neobert-vendored")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="converted NeoBERT checkpoint not present")


def test_strict_load_no_missing_unexpected():
    _, info = NeoBERTForMaskedLM.from_pretrained(CKPT, output_loading_info=True)
    assert not info["missing_keys"], info["missing_keys"]
    assert not info["unexpected_keys"], info["unexpected_keys"]


def test_save_reload_roundtrip(tmp_path):
    model = NeoBERTForMaskedLM.from_pretrained(CKPT)
    sd0 = model.state_dict()
    model.save_pretrained(tmp_path)
    reloaded = NeoBERTForMaskedLM.from_pretrained(tmp_path)
    sd1 = reloaded.state_dict()
    assert set(sd0) == set(sd1)
    for k in sd0:
        assert torch.equal(sd0[k], sd1[k]), k
    # canary for the F2b guarded-init trap and tie-machinery (both silent under strict load)
    assert not torch.equal(reloaded.decoder.weight, reloaded.model.encoder.weight)


def test_decoder_untied():
    model = NeoBERTForMaskedLM.from_pretrained(CKPT)
    assert not torch.equal(model.decoder.weight, model.model.encoder.weight)
