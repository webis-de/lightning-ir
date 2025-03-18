from transformers.models.bert.modeling_bert import BertOnlyMLMHead

MODEL_TYPE_TO_LM_HEAD = {
    "bert": BertOnlyMLMHead,
}

MODEL_TYPE_TO_HEAD_NAME = {
    "bert": "cls",
}

# NOTE: In the output embeddings and tied weight keys the cls key has already been unified and replaced by the
# projection key

MODEL_TYPE_TO_OUTPUT_EMBEDDINGS = {
    "bert": "projection.predictions.decoder",
}

MODEL_TYPE_TO_TIED_WEIGHTS_KEYS = {
    "bert": ["projection.predictions.decoder.bias", "projection.predictions.decoder.weight"],
}
