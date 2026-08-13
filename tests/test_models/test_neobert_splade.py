"""SPLADE-over-NeoBERT integration tests.

SPLADE is the one lightning-ir model type that needs more than the backbone: it re-creates the
backbone's pre-trained MLM head as ``model.projection`` and loads the head weights out of the
checkpoint via ``key_mapping``. These tests pin the three things that can silently go wrong:

- the head *shape* (NeoBERT's MLM head is a bare vocabulary projection, not BERT's dense/act/norm
  transform + decoder), checked by exact logit parity against ``NeoBERTForMaskedLM``;
- the head *weights* actually coming from the checkpoint rather than being newly initialized;
- the head *not* being tied to the input embeddings — NeoBERT pre-trains an untied decoder
  (``tie_word_embeddings=False``), and tying would overwrite it with the word embeddings.

CPU, CI-safe: skipped unless the converted checkpoint exists.
"""

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import lightning_ir  # noqa: F401  triggers _register_backbones()
from lightning_ir import BiEncoderModule
from lightning_ir.modeling_utils.lm_head import LinearLMHead
from lightning_ir.models import SpladeConfig
from lightning_ir.models.backbones.neobert import NeoBERTForMaskedLM

CKPT = Path("checkpoints/neobert-vendored")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="converted NeoBERT checkpoint not present")


def _splade_module(**config_kwargs):
    return BiEncoderModule(model_name_or_path=str(CKPT), config=SpladeConfig(**config_kwargs)).eval()


def _doc_encoding(module, text="Paris is the capital of France."):
    return module.tokenizer.tokenize(docs=[text], return_tensors="pt", padding=True)["doc_encoding"]


def test_factory_derivation_and_head_shape():
    """The factory builds SpladeNeoBERTModel with a transform-free vocabulary projection."""
    model = _splade_module().model
    assert type(model).__name__ == "SpladeNeoBERTModel"
    assert isinstance(model.projection, LinearLMHead)
    # NeoBERT's head has no dense/act/norm block — a BERT-style LMHead here would be wrong weights.
    assert not hasattr(model.projection, "dense")
    assert model.projection.decoder.out_features == model.config.vocab_size


def test_projection_is_the_pretrained_mlm_head():
    """Exact logit parity with NeoBERTForMaskedLM: right shape, right weights, right input."""
    module = _splade_module()
    mlm = NeoBERTForMaskedLM.from_pretrained(str(CKPT)).eval()
    encoding = _doc_encoding(module)
    with torch.inference_mode():
        reference = mlm(**encoding).logits
        ours = module.model.projection(module.model._backbone_forward(**encoding).last_hidden_state)
    assert torch.equal(reference, ours)


def test_head_weights_come_from_the_checkpoint():
    """The key_mapping lands ``decoder.*`` on ``projection.decoder.*`` rather than re-initializing it."""
    decoder = _splade_module().model.projection.decoder
    with safe_open(CKPT / "model.safetensors", "pt") as f:
        assert torch.equal(decoder.weight.data, f.get_tensor("decoder.weight"))
        assert torch.equal(decoder.bias.data, f.get_tensor("decoder.bias"))


def test_decoder_not_tied_to_input_embeddings():
    """NeoBERT pre-trains an untied MLM decoder; tying would clobber it with the word embeddings."""
    model = _splade_module().model
    assert model.config.tie_word_embeddings is False
    assert "projection.decoder.weight" not in model.all_tied_weights_keys
    embeddings = model.get_input_embeddings().weight
    assert model.projection.decoder.weight.data_ptr() != embeddings.data_ptr()
    assert not torch.equal(model.projection.decoder.weight.data, embeddings.data)


def test_forward_is_finite_and_sparse_valued():
    module = _splade_module()
    with torch.inference_mode():
        output = module.score("capital of France?", ["Paris is the capital.", "Bananas are yellow."])
    for embeddings in (output.query_embeddings.embeddings, output.doc_embeddings.embeddings):
        assert torch.isfinite(embeddings).all()
        assert embeddings.shape[-1] == module.model.config.vocab_size
        assert (embeddings >= 0).all()  # relu_log sparsification
    assert torch.isfinite(output.scores).all()


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {},
        {"query_weighting": "static", "query_expansion": False},  # inference-free query encoding
        {"query_weighting": None, "query_expansion": False},  # lexical-only query encoding
    ],
)
def test_gradients_flow(config_kwargs):
    module = _splade_module(**config_kwargs)
    module.train()
    encoding = module.tokenizer.tokenize(
        queries=["capital of France?"],
        docs=["Paris is the capital.", "Bananas are yellow."],
        return_tensors="pt",
        padding=True,
    )
    output = module.model.forward(encoding["query_encoding"], encoding["doc_encoding"], num_docs=[2])
    output.scores.sum().backward()
    assert module.model.projection.decoder.weight.grad is not None
    assert torch.isfinite(module.model.projection.decoder.weight.grad).all()


def test_reload_preserves_head(tmp_path):
    """The untied decoder must survive save -> reload (a registered tie would drop it on save)."""
    module = _splade_module()
    before = {k: v.clone() for k, v in module.model.state_dict().items()}
    module.model.save_pretrained(tmp_path)
    module.tokenizer.save_pretrained(tmp_path)
    after = BiEncoderModule(model_name_or_path=str(tmp_path)).model.state_dict()
    assert "projection.decoder.weight" in after
    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"tensor changed on reload: {key}"
