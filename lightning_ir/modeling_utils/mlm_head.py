import torch
from transformers.activations import get_activation
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig


class DistilBertOnlyMLMHead(torch.nn.Module):
    def __init__(self, config: DistilBertConfig) -> None:
        super().__init__()
        self.activation = get_activation(config.activation)
        self.vocab_transform = torch.nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = torch.nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = torch.nn.Linear(config.dim, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vocab_transform(x)
        x = self.activation(x)
        x = self.vocab_layer_norm(x)
        x = self.vocab_projector(x)
        return x


MODEL_TYPE_TO_LM_HEAD = {
    "bert": BertOnlyMLMHead,
    "distilbert": DistilBertOnlyMLMHead,
}

MODEL_TYPE_TO_STATE_DICT_KEY_MAPPING = {
    "bert": {"cls": "bert.projection"},
    "distilbert": {
        "vocab_transform": "distilbert.projection.vocab_transform",
        "vocab_layer_norm": "distilbert.projection.vocab_layer_norm",
        "vocab_projector": "distilbert.projection.vocab_projector",
    },
}

# NOTE: In the output embeddings and tied weight keys the cls key has already been unified and replaced by the
# projection key

MODEL_TYPE_TO_OUTPUT_EMBEDDINGS = {
    "bert": "predictions.decoder",
    "distilbert": "vocab_projector",
}

MODEL_TYPE_TO_TIED_WEIGHTS_KEYS = {
    "bert": ["predictions.decoder.bias", "predictions.decoder.weight"],
    "distilbert": ["vocab_projector.bias", "vocab_projector.weight"],
}
