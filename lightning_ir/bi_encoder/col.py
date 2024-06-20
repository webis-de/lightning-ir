from __future__ import annotations

import json
from pathlib import Path
from string import punctuation
from typing import Tuple

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)


from .config import MultiVectorBiEncoderConfig
from .model import MultiVectorBiEncoderModel
from .tokenizer import BiEncoderTokenizer


class ColConfig(MultiVectorBiEncoderConfig):
    model_type = "col"

    ADDED_ARGS = MultiVectorBiEncoderConfig.ADDED_ARGS + ["mask_punctuation"]

    def __init__(self, mask_punctuation: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_punctuation = mask_punctuation


class ColModel(MultiVectorBiEncoderModel):
    config_class = ColConfig
    tokenizer_class = BiEncoderTokenizer

    def __init__(self) -> None:
        # TODO move token masking to multi-vector-bi-encoder
        super().__init__()
        self.config: ColConfig
        self.mask_tokens = None
        if self.config.mask_punctuation:
            try:
                tokenizer = BiEncoderTokenizer.from_pretrained(self.config.name_or_path)
            except OSError:
                raise ValueError(
                    "Can't use mask_punctuation if the checkpoint does not "
                    "have a tokenizer."
                )
            self.mask_tokens = tokenizer(
                punctuation,
                add_special_tokens=False,
                return_tensors="pt",
                return_attention_mask=False,
                return_token_type_ids=False,
                warn=False,
            ).input_ids[0]

    def scoring_masks(
        self,
        query_input_ids: torch.Tensor,
        doc_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor | None,
        doc_attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_scoring_mask, doc_scoring_mask = super().scoring_masks(
            query_input_ids, doc_input_ids, query_attention_mask, doc_attention_mask
        )
        if self.mask_tokens is not None:
            punctuation_mask = (
                doc_input_ids[..., None]
                .eq(self.mask_tokens.to(doc_attention_mask))
                .any(-1)
            )
            doc_scoring_mask = doc_scoring_mask & ~punctuation_mask
        return query_scoring_mask, doc_scoring_mask

    @classmethod
    def from_col_checkpoint(cls, model_name_or_path: Path | str) -> "ColModel":
        col_config = None
        if isinstance(model_name_or_path, Path):
            if (model_name_or_path / "artifact.metadata").exists():
                col_config = json.loads(
                    (model_name_or_path / "artifact.metadata").read_text()
                )
        try:
            col_config_path = hf_hub_download(
                repo_id=str(model_name_or_path), filename="artifact.metadata"
            )
            col_config = json.loads(Path(col_config_path).read_text())
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if col_config is None:
            raise ValueError("{model_name_or_path} is not a valid col checkpoint.")
        config = ColConfig.from_pretrained(model_name_or_path)
        config.update(
            {
                "similarity_function": "dot",
                "aggregation_function": "sum",
                "query_expansion": True,
                "query_length": col_config["query_maxlen"],
                "attend_to_query_expanded_tokens": col_config["attend_to_mask_tokens"],
                "doc_expansion": False,
                "doc_length": 512,
                "attend_to_doc_expanded_tokens": False,
                "normalize": True,
                "add_marker_tokens": True,
                "embedding_dim": col_config["dim"],
                "mask_punctuation": col_config["mask_punctuation"],
                "linear_bias": False,
            }
        )
        model = cls.from_pretrained(model_name_or_path, config=config)
        query_token_id = config.vocab_size
        doc_token_id = config.vocab_size + 1
        model.resize_token_embeddings(config.vocab_size + 2, 8)
        embeddings = model.bert.embeddings.word_embeddings.weight.data
        embeddings[query_token_id] = embeddings[tokenizer.vocab["[unused0]"]]
        embeddings[doc_token_id] = embeddings[tokenizer.vocab["[unused1]"]]
        return model


AutoConfig.register(ColConfig.model_type, ColConfig)
AutoModel.register(ColConfig, ColModel)
