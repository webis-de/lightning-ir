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
    "roberta": partial(LMHead, hidden_dim_key="hidden_size", activation_key="hidden_act"),
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
    "roberta": {
        "lm_head.dense": "roberta.projection.dense",
        "lm_head.layer_norm": "roberta.projection.norm",
        "lm_head.decoder": "roberta.projection.decoder",
        "lm_head.bias": "roberta.projection.decoder.bias",
    },
}

# Maps backbone model_type -> path to input embedding weight (for tied weights in v5)
MODEL_TYPE_TO_EMBEDDING_WEIGHT_KEY = {
    "bert": "bert.embeddings.word_embeddings.weight",
    "distilbert": "distilbert.embeddings.word_embeddings.weight",
    "modernbert": "model.embeddings.tok_embeddings.weight",
    "roberta": "roberta.embeddings.word_embeddings.weight",
}
