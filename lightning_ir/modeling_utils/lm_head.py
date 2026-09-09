from functools import partial

import torch
from transformers import PretrainedConfig
from transformers.activations import get_activation


class LMHead(torch.nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_dim_key: str,
        activation_key: str,
        classifier_bias_key: str | None = None,
        norm_bias_key: str | None = None,
    ):
        super().__init__()
        dim = getattr(config, hidden_dim_key)
        activation = getattr(config, activation_key)
        classifier_bias = True if classifier_bias_key is None else getattr(config, classifier_bias_key, True)
        norm_bias = True if norm_bias_key is None else getattr(config, norm_bias_key, True)
        self.dense = torch.nn.Linear(dim, dim, bias=classifier_bias)
        self.act = get_activation(activation)
        self.norm = torch.nn.LayerNorm(dim, eps=1e-12, bias=norm_bias)
        self.decoder = torch.nn.Linear(dim, config.vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(hidden_states))))


class LinearLMHead(torch.nn.Module):
    """Transform-free masked language modeling head: a single vocabulary projection.

    Used by backbones whose pre-trained MLM head is a bare ``nn.Linear`` over the final hidden state,
    with no dense/activation/norm transform block. NeoBERT is such a backbone — its encoder already
    applies a final ``RMSNorm``, and ``NeoBERTForMaskedLM`` projects that output straight to the
    vocabulary. The ``decoder`` attribute name matches :class:`LMHead` so that
    :meth:`~lightning_ir.models.bi_encoders.splade.SpladeModel.get_output_embeddings` and the
    checkpoint key mappings below work unchanged.
    """

    def __init__(self, config: PretrainedConfig, hidden_dim_key: str = "hidden_size"):
        super().__init__()
        self.decoder = torch.nn.Linear(getattr(config, hidden_dim_key), config.vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden_states)


MODEL_TYPE_TO_LM_HEAD = {
    "bert": partial(LMHead, hidden_dim_key="hidden_size", activation_key="hidden_act"),
    "distilbert": partial(LMHead, hidden_dim_key="hidden_size", activation_key="activation"),
    "modernbert": partial(
        LMHead,
        hidden_dim_key="hidden_size",
        activation_key="classifier_activation",
        classifier_bias_key="classifier_bias",
        norm_bias_key="norm_bias",
    ),
    # NeoBERT's pre-trained MLM head is a plain vocabulary projection (no transform block).
    "neobert": partial(LinearLMHead, hidden_dim_key="hidden_size"),
    "roberta": partial(LMHead, hidden_dim_key="hidden_size", activation_key="hidden_act"),
    # DeBERTa(-v2/-v3). Structurally BERT-shaped (dense -> act -> LayerNorm -> tied decoder),
    # so no new head class is needed. BUT: the released microsoft/deberta-v3-* checkpoints do
    # NOT carry a usable pre-trained MLM head -- see the note on the key mapping below. SPLADE
    # over this backbone therefore starts from an effectively randomly-initialised vocabulary
    # projection, unlike every other backbone in the grid.
    "deberta-v2": partial(LMHead, hidden_dim_key="hidden_size", activation_key="hidden_act"),
}

MODEL_TYPE_TO_STATE_DICT_KEY_MAPPING = {
    "bert": {
        "cls.predictions.transform.dense": "bert.projection.dense",
        "cls.predictions.transform.LayerNorm": "bert.projection.norm",
        "cls.predictions.decoder": "bert.projection.decoder",
        "cls.predictions.bias": "bert.projection.decoder.bias",
    },
    "distilbert": {
        "vocab_transform": "distilbert.projection.dense",
        "vocab_layer_norm": "distilbert.projection.norm",
        "vocab_projector": "distilbert.projection.decoder",
    },
    "modernbert": {
        "head.dense": "model.projection.dense",
        "head.norm": "model.projection.norm",
        "decoder": "model.projection.decoder",
    },
    # ``NeoBERTForMaskedLM`` keeps the encoder under ``model.`` and the MLM projection at the top
    # level (``decoder.weight`` / ``decoder.bias``), same layout as ModernBERT. Re-homing it under
    # the ``model.`` prefix lets the usual base-model-prefix stripping land it on
    # ``projection.decoder``. Anchored so it cannot match ``model.transformer_encoder.*``.
    "neobert": {
        r"^decoder\.": "model.projection.decoder.",
    },
    # DeBERTa-v3 (model_type "deberta-v2"). The checkpoint carries `lm_predictions.lm_head.*`,
    # left over from the ELECTRA-style RTD pre-training in which the released base model is the
    # DISCRIMINATOR (`mask_predictions.classifier` is (1, 768), a binary real/fake head) and the
    # MLM head belonged to the discarded generator.
    #
    # `dense.weight` is DELIBERATELY NOT MAPPED. Measured on microsoft/deberta-v3-base:
    #   * the checkpoint's copy has std 0.000994 -- two orders below a trained 768x768 layer;
    #   * injecting it by hand into DebertaV2ForMaskedLM leaves masked-token prediction as
    #     nonsense ("The capital of France is [MASK]." -> Brack, caine, aksh, Callan), i.e. no
    #     better than the random init transformers falls back to.
    # So a "correct" mapping does not rescue this head, and loading a dead matrix is worse than
    # a fresh one. LayerNorm and the vocabulary bias ARE mapped: they are real trained statistics
    # (the vocab bias encodes token-frequency priors, which SPLADE can use).
    "deberta-v2": {
        "lm_predictions.lm_head.LayerNorm": "deberta.projection.norm",
        "lm_predictions.lm_head.bias": "deberta.projection.decoder.bias",
    },
    "roberta": {
        "lm_head.dense": "roberta.projection.dense",
        "lm_head.layer_norm": "roberta.projection.norm",
        "lm_head.decoder": "roberta.projection.decoder",
        "lm_head.bias": "roberta.projection.decoder.bias",
    },
}

# Path (relative to the backbone) of the input word-embedding weight that the SPLADE
# MLM projection decoder is tied to. ModernBERT names this ``tok_embeddings`` instead of
# the ``word_embeddings`` used by the BERT-family encoders; NeoBERT's word embedding is a
# bare ``nn.Embedding`` named ``encoder``.
#
# Only consulted for backbones that actually tie the two (``config.tie_word_embeddings``);
# NeoBERT pre-trains an *untied* decoder, so the entry below is unused in practice.
MODEL_TYPE_TO_INPUT_EMBEDDINGS_KEY = {
    "bert": "embeddings.word_embeddings.weight",
    # deberta-v3 ties the decoder to the word embeddings (config.tie_word_embeddings=True),
    # so this IS consulted -- and the tie is the only genuinely pre-trained part of the
    # SPLADE projection for this backbone.
    "deberta-v2": "embeddings.word_embeddings.weight",
    "distilbert": "embeddings.word_embeddings.weight",
    "modernbert": "embeddings.tok_embeddings.weight",
    "neobert": "encoder.weight",
    "roberta": "embeddings.word_embeddings.weight",
}
