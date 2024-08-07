import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers.modeling_utils import load_state_dict

from ...base import LightningIRModel, LightningIRModelClassFactory, LightningIRTokenizerClassFactory
from ...bi_encoder.model import BiEncoderModel
from .config import ColConfig


def round_to_multiple_of_8(x: int) -> int:
    return ((x + 7) // 8) * 8


class ColModel(BiEncoderModel):
    config_class = ColConfig

    def __init__(self, config: ColConfig, *args, **kwargs) -> None:
        super().__init__(config)
        self.config: ColConfig

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path, *args, **kwargs) -> LightningIRModel:
        try:
            hf_hub_download(repo_id=str(model_name_or_path), filename="artifact.metadata")
        except Exception:
            return super().from_pretrained(model_name_or_path, *args, **kwargs)
        return cls.from_colbert_checkpoint(model_name_or_path)

    @classmethod
    def from_colbert_checkpoint(cls, model_name_or_path: Path | str) -> "ColModel":
        col_config = None
        try:
            col_config_path = hf_hub_download(repo_id=str(model_name_or_path), filename="artifact.metadata")
            col_config = json.loads(Path(col_config_path).read_text())
        except Exception:
            pass
        if col_config is None:
            raise ValueError(f"{model_name_or_path} is not a valid col checkpoint.")
        cls = LightningIRModelClassFactory(ColConfig).from_pretrained(model_name_or_path)
        config = cls.config_class.from_pretrained(model_name_or_path)
        config.update(
            {
                "name_or_path": str(Path(col_config_path).parent),
                "similarity_function": "dot",
                "query_aggregation_function": "sum",
                "query_expansion": True,
                "query_length": col_config["query_maxlen"],
                "attend_to_query_expanded_tokens": col_config["attend_to_mask_tokens"],
                "doc_expansion": False,
                "doc_length": round_to_multiple_of_8(col_config["doc_maxlen"]),
                "attend_to_doc_expanded_tokens": False,
                "normalize": True,
                "add_marker_tokens": True,
                "embedding_dim": col_config["dim"],
                "doc_mask_scoring_tokens": ("punctuation" if col_config["mask_punctuation"] else None),
            }
        )
        model = cls(config=config)
        state_dict_path = hf_hub_download(repo_id=str(model_name_or_path), filename="model.safetensors")
        state_dict = load_state_dict(state_dict_path)
        state_dict.pop("bert.embeddings.position_ids", None)
        for key in list(state_dict.keys()):
            if key.startswith("bert."):
                state_dict[key[5:]] = state_dict.pop(key)
        state_dict["projection.weight"] = state_dict.pop("linear.weight")
        model.load_state_dict(state_dict)

        tokenizer = (
            LightningIRTokenizerClassFactory(ColConfig)
            .from_pretrained(model_name_or_path)
            .from_pretrained(model_name_or_path, config=config)
        )
        query_token_id = tokenizer.query_token_id
        doc_token_id = tokenizer.doc_token_id
        model.resize_token_embeddings(len(tokenizer), 8)
        embeddings = model.embeddings.word_embeddings.weight.data
        embeddings[query_token_id] = embeddings[tokenizer.vocab["[unused0]"]]
        embeddings[doc_token_id] = embeddings[tokenizer.vocab["[unused1]"]]
        return model
