import warnings

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import T5EncoderModel

from ..base import BACKBONE_MAPPING, CHECKPOINT_MAPPING, POST_LOAD_CALLBACKS, STATE_DICT_KEY_MAPPING, LightningIRModel
from ..models import CoilConfig, ColConfig, DprConfig, MonoConfig, SpladeConfig, UniCoilConfig, XTRConfig


def _map_colbert_marker_tokens(model: LightningIRModel) -> LightningIRModel:
    config = model.config
    query_token_id = config.vocab_size
    doc_token_id = config.vocab_size + 1
    model.resize_token_embeddings(config.vocab_size + 2, 8)
    embeddings = model.embeddings.word_embeddings.weight.data
    embeddings[query_token_id] = embeddings[1]  # [unused0]
    embeddings[doc_token_id] = embeddings[2]  # [unused1]
    return model


def _map_moderncolbert_marker_tokens(model: LightningIRModel) -> LightningIRModel:
    config = model.config
    query_token_id = config.vocab_size
    doc_token_id = config.vocab_size + 1
    model.resize_token_embeddings(config.vocab_size + 2, 8)
    embeddings = model.embeddings.tok_embeddings.weight.data
    embeddings[query_token_id] = embeddings[50368]  # [unused0]
    embeddings[doc_token_id] = embeddings[50369]  # [unused1]

    path = hf_hub_download(model.config.name_or_path, filename="model.safetensors", subfolder="1_Dense")
    state_dict = load_file(path)
    state_dict["weight"] = state_dict.pop("linear.weight")
    model.projection.load_state_dict(state_dict)
    return model


def _map_xtr_weights(model: LightningIRModel) -> LightningIRModel:
    path = hf_hub_download(model.config.name_or_path, filename="pytorch_model.bin", subfolder="2_Dense")
    state_dict = torch.load(path, weights_only=True, map_location="cpu")
    state_dict["projection.weight"] = state_dict.pop("linear.weight")
    model.load_state_dict(state_dict, strict=False)
    return model


def _map_mono_t5_weights(model: LightningIRModel) -> LightningIRModel:
    # [1176, 6136] true, false
    warnings.warn(
        "The above warning, that the linear layer is not initialized, is expected and can be ignored."
        "The weights are initialized separately."
    )
    model.linear.weight.data = model.shared.weight.data[[6136, 1176]]
    return model


def _map_rank_t5_weights(model: LightningIRModel) -> LightningIRModel:
    # 32089 <extra_id_10>
    warnings.warn(
        "The above warning, that the linear layer is not initialized, is expected and can be ignored."
        "The weights are initialized separately."
    )
    model.linear.weight.data = model.shared.weight.data[[32089]]
    return model


def _map_coil_weights(model: LightningIRModel) -> LightningIRModel:
    path = hf_hub_download(model.config.name_or_path, filename="model.pt")
    state_dict = torch.load(path, map_location="cpu")
    state_dict["token_projection.weight"] = state_dict.pop("tok_proj.weight")
    state_dict["token_projection.bias"] = state_dict.pop("tok_proj.bias")
    state_dict["cls_projection.weight"] = state_dict.pop("cls_proj.weight")
    state_dict["cls_projection.bias"] = state_dict.pop("cls_proj.bias")
    model.load_state_dict(state_dict, strict=False)
    return model


def _map_opensearch_splade_weights(model: LightningIRModel) -> LightningIRModel:
    path = hf_hub_download(
        model.config.name_or_path, filename="model.safetensors", subfolder="query_0_SparseStaticEmbedding"
    )
    state_dict = load_file(path)
    state_dict["weight"] = state_dict.pop("weight").unsqueeze(-1)
    model.query_weights.load_state_dict(state_dict)
    return model


MONO_T5_PATTERN = "Query: {query} Document: {doc} Relevant:"
RANK_T5_PATTERN = "Query: {query} Document: {doc}"


def _register_external_models():
    CHECKPOINT_MAPPING.update(
        {
            "colbert-ir/colbertv2.0": ColConfig(
                query_length=32,
                doc_length=184,
                add_marker_tokens=True,
                normalization="l2",
                query_expansion=True,
                doc_mask_scoring_tokens="punctuation",
            ),
            "lightonai/GTE-ModernColBERT-v1": ColConfig(
                query_length=32,
                doc_length=296,
                add_marker_tokens=True,
                normalization="l2",
                query_expansion=False,
                projection="linear_no_bias",
                doc_mask_scoring_tokens="punctuation",
            ),
            "naver/splade-v3": SpladeConfig(),
            "naver/splade-v3-distilbert": SpladeConfig(),
            "naver/splade-v3-doc": SpladeConfig(query_expansion=False, query_weighting=None),
            "naver/splade-v3-lexical": SpladeConfig(query_expansion=False),
            "naver/splade_v2_distil": SpladeConfig(),
            "opensearch-project/opensearch-neural-sparse-encoding-v2-distill": SpladeConfig(),
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill": SpladeConfig(
                query_expansion=False, query_weighting="static"
            ),
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini": SpladeConfig(
                query_expansion=False,
                query_weighting="static",
            ),
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill": SpladeConfig(
                query_expansion=False, query_weighting="static", sparsification="relu_2xlog"
            ),
            "sentence-transformers/msmarco-bert-base-dot-v5": DprConfig(
                projection=None, query_pooling_strategy="mean", doc_pooling_strategy="mean"
            ),
            "sentence-transformers/msmarco-distilbert-dot-v5": DprConfig(
                projection=None, query_pooling_strategy="mean", doc_pooling_strategy="mean"
            ),
            "sentence-transformers/msmarco-MiniLM-L-6-v3": DprConfig(
                projection=None, query_pooling_strategy="mean", doc_pooling_strategy="mean"
            ),
            "castorini/monot5-base-msmarco-10k": MonoConfig(scoring_strategy="mono", tokenizer_pattern=MONO_T5_PATTERN),
            "castorini/monot5-base-msmarco": MonoConfig(scoring_strategy="mono", tokenizer_pattern=MONO_T5_PATTERN),
            "castorini/monot5-large-msmarco-10k": MonoConfig(
                scoring_strategy="mono", tokenizer_pattern=MONO_T5_PATTERN
            ),
            "castorini/monot5-large-msmarco": MonoConfig(scoring_strategy="mono", tokenizer_pattern=MONO_T5_PATTERN),
            "castorini/monot5-3b-msmarco-10k": MonoConfig(scoring_strategy="mono", tokenizer_pattern=MONO_T5_PATTERN),
            "castorini/monot5-3b-msmarco": MonoConfig(scoring_strategy="mono", tokenizer_pattern=MONO_T5_PATTERN),
            "Soyoung97/RankT5-base": MonoConfig(scoring_strategy="rank", tokenizer_pattern=RANK_T5_PATTERN),
            "Soyoung97/RankT5-large": MonoConfig(scoring_strategy="rank", tokenizer_pattern=RANK_T5_PATTERN),
            "Soyoung97/RankT5-3b": MonoConfig(scoring_strategy="rank", tokenizer_pattern=RANK_T5_PATTERN),
            "castorini/monobert-large-msmarco-finetune-only": MonoConfig(
                scoring_strategy="mono", linear_bias=True, pooling_strategy="bert_pool"
            ),
            "castorini/monobert-large-msmarco": MonoConfig(
                scoring_strategy="mono", linear_bias=True, pooling_strategy="bert_pool"
            ),
            "fschlatt/coil-with-hn": CoilConfig(),
            "castorini/unicoil-noexp-msmarco-passage": UniCoilConfig(projection="linear"),
            "google/xtr-base-en": XTRConfig(projection="linear_no_bias"),
        }
    )
    BACKBONE_MAPPING.update({"google/xtr-base-en": T5EncoderModel})
    STATE_DICT_KEY_MAPPING.update(
        {
            "colbert-ir/colbertv2.0": [("linear.weight", "bert.projection.weight")],
            "castorini/monobert-large-msmarco-finetune-only": [
                ("classifier.weight", "bert.linear.weight"),
                ("classifier.bias", "bert.linear.bias"),
                ("bert.pooler.dense.weight", "bert.bert_pool.0.weight"),
                ("bert.pooler.dense.bias", "bert.bert_pool.0.bias"),
            ],
            "castorini/monobert-large-msmarco": [
                ("classifier.weight", "bert.linear.weight"),
                ("classifier.bias", "bert.linear.bias"),
                ("bert.pooler.dense.weight", "bert.bert_pool.0.weight"),
                ("bert.pooler.dense.bias", "bert.bert_pool.0.bias"),
            ],
            "castorini/unicoil-noexp-msmarco-passage": [
                ("coil_encoder.bert.", ""),
                ("coil_encoder.tok_proj", "token_projection"),
            ],
        }
    )
    POST_LOAD_CALLBACKS.update(
        {
            "colbert-ir/colbertv2.0": _map_colbert_marker_tokens,
            "lightonai/GTE-ModernColBERT-v1": _map_moderncolbert_marker_tokens,
            "castorini/monot5-base-msmarco-10k": _map_mono_t5_weights,
            "castorini/monot5-base-msmarco": _map_mono_t5_weights,
            "castorini/monot5-large-msmarco-10k": _map_mono_t5_weights,
            "castorini/monot5-large-msmarco": _map_mono_t5_weights,
            "castorini/monot5-3b-msmarco-10k": _map_mono_t5_weights,
            "castorini/monot5-3b-msmarco": _map_mono_t5_weights,
            "fschlatt/coil-with-hn": _map_coil_weights,
            "google/xtr-base-en": _map_xtr_weights,
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill": _map_opensearch_splade_weights,
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini": _map_opensearch_splade_weights,
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill": _map_opensearch_splade_weights,
            "Soyoung97/RankT5-base": _map_rank_t5_weights,
            "Soyoung97/RankT5-large": _map_rank_t5_weights,
            "Soyoung97/RankT5-3b": _map_rank_t5_weights,
        }
    )
