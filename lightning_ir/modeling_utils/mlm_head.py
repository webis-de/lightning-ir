from transformers.models.bert.modeling_bert import BertOnlyMLMHead

MODEL_TYPE_TO_LM_HEAD = {
    "bert": BertOnlyMLMHead,
}

MODEL_TYPE_TO_KEY_MAPPING = {
    "bert": {"cls": "bert.projection"},
}

# NOTE: In the output embeddings and tied weight keys the cls key has already been unified and replaced by the
# projection key

MODEL_TYPE_TO_OUTPUT_EMBEDDINGS = {
    "bert": "predictions.decoder",
}

MODEL_TYPE_TO_TIED_WEIGHTS_KEYS = {
    "bert": ["predictions.decoder.bias", "predictions.decoder.weight"],
}
