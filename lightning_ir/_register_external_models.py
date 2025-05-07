import warnings

from .base import CHECKPOINT_MAPPING, POST_LOAD_CALLBACKS, STATE_DICT_KEY_MAPPING, LightningIRModel
from .models import ColConfig, DprConfig, SpladeConfig, T5CrossEncoderConfig


def _map_colbert_marker_tokens(model: LightningIRModel) -> LightningIRModel:
    config = model.config
    query_token_id = config.vocab_size
    doc_token_id = config.vocab_size + 1
    model.resize_token_embeddings(config.vocab_size + 2, 8)
    embeddings = model.embeddings.word_embeddings.weight.data
    embeddings[query_token_id] = embeddings[1]  # [unused0]
    embeddings[doc_token_id] = embeddings[2]  # [unused1]
    return model


def _map_mono_t5_weights(model: LightningIRModel) -> LightningIRModel:
    # [1176, 6136] true, false
    warnings.warn(
        "The above warning, that the linear layer is not initialized, is expected and can be ignored."
        "The weights are initialized separately."
    )
    model.linear.weight.data = model.shared.weight.data[[1176, 6136]]
    return model


def _map_rank_t5_weights(model: LightningIRModel) -> LightningIRModel:
    # 32089 <extra_id_10>
    warnings.warn(
        "The above warning, that the linear layer is not initialized, is expected and can be ignored."
        "The weights are initialized separately."
    )
    model.linear.weight.data = model.shared.weight.data[[32089]]
    return model


def _register_external_models():
    CHECKPOINT_MAPPING.update(
        {
            "colbert-ir/colbertv2.0": ColConfig(
                query_length=32,
                doc_length=184,
                add_marker_tokens=True,
                normalize=True,
                query_expansion=True,
                doc_mask_scoring_tokens="punctuation",
            ),
            "naver/splade-v3": SpladeConfig(),
            "sentence-transformers/msmarco-bert-base-dot-v5": DprConfig(
                projection=None, query_pooling_strategy="mean", doc_pooling_strategy="mean"
            ),
            "sentence-transformers/msmarco-distilbert-dot-v5": DprConfig(
                projection=None, query_pooling_strategy="mean", doc_pooling_strategy="mean"
            ),
            "sentence-transformers/msmarco-MiniLM-L-6-v3": DprConfig(
                projection=None, query_pooling_strategy="mean", doc_pooling_strategy="mean"
            ),
            "castorini/monot5-base-msmarco-10k": T5CrossEncoderConfig(decoder_strategy="mono"),
            "castorini/monot5-base-msmarco": T5CrossEncoderConfig(decoder_strategy="mono"),
            "castorini/monot5-large-msmarco-10k": T5CrossEncoderConfig(decoder_strategy="mono"),
            "castorini/monot5-large-msmarco": T5CrossEncoderConfig(decoder_strategy="mono"),
            "castorini/monot5-3b-msmarco-10k": T5CrossEncoderConfig(decoder_strategy="mono"),
            "castorini/monot5-3b-msmarco": T5CrossEncoderConfig(decoder_strategy="mono"),
            "Soyoung97/RankT5-base": T5CrossEncoderConfig(decoder_strategy="rank"),
            "Soyoung97/RankT5-large": T5CrossEncoderConfig(decoder_strategy="rank"),
            "Soyoung97/RankT5-3b": T5CrossEncoderConfig(decoder_strategy="rank"),
        }
    )
    STATE_DICT_KEY_MAPPING.update(
        {
            "colbert-ir/colbertv2.0": [("linear.weight", "bert.projection.weight")],
        }
    )
    POST_LOAD_CALLBACKS.update(
        {
            "colbert-ir/colbertv2.0": _map_colbert_marker_tokens,
            "castorini/monot5-base-msmarco-10k": _map_mono_t5_weights,
            "castorini/monot5-base-msmarco": _map_mono_t5_weights,
            "castorini/monot5-large-msmarco-10k": _map_mono_t5_weights,
            "castorini/monot5-large-msmarco": _map_mono_t5_weights,
            "castorini/monot5-3b-msmarco-10k": _map_mono_t5_weights,
            "castorini/monot5-3b-msmarco": _map_mono_t5_weights,
            "Soyoung97/RankT5-base": _map_rank_t5_weights,
            "Soyoung97/RankT5-large": _map_rank_t5_weights,
            "Soyoung97/RankT5-3b": _map_rank_t5_weights,
        }
    )
