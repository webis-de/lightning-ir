import json
from pathlib import Path
from string import punctuation
from typing import Any, Dict

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BertConfig, BertModel, BertPreTrainedModel

from .flash.flash_model import FlashClassFactory
from .loss import LossFunction
from .mvr import MVRConfig, MVRModel, MVRModule, MVRTokenizer


class ColBERTConfig(BertConfig, MVRConfig):

    model_type = "colbert"

    def __init__(self, mask_punctuation: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_punctuation = mask_punctuation

    def to_mvr_dict(self) -> Dict[str, Any]:
        mvr_dict = super().to_mvr_dict()
        mvr_dict["mask_punctuation"] = self.mask_punctuation
        return mvr_dict


class ColBERTModel(BertPreTrainedModel, MVRModel):
    def __init__(self, colbert_config: ColBERTConfig) -> None:
        super().__init__(colbert_config)
        self.bert = BertModel(colbert_config, add_pooling_layer=False)
        self.linear = torch.nn.Linear(
            colbert_config.hidden_size,
            colbert_config.embedding_dim,
            bias=self.config.linear_bias,
        )
        self.mask_tokens = None
        if self.config.mask_punctuation:
            try:
                tokenizer = MVRTokenizer.from_pretrained(self.config.name_or_path)
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
            ).input_ids[0]
        self.post_init()

    @property
    def encoder(self) -> torch.nn.Module:
        return self.bert

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None = None,
    ):
        embedding = super().encode_docs(input_ids, attention_mask, token_type_ids)
        if self.mask_tokens is not None:
            punctuation_mask = (
                input_ids[..., None].eq(self.mask_tokens.to(attention_mask)).any(-1)
            )
            embedding[punctuation_mask] = self.scoring_function.MASK_VALUE
        return embedding

    @classmethod
    def from_colbert_checkpoint(cls, model_name_or_path: Path | str) -> "ColBERTModel":
        colbert_config = None
        if isinstance(model_name_or_path, Path):
            if (model_name_or_path / "artifact.metadata").exists():
                colbert_config = json.loads(
                    (model_name_or_path / "artifact.metadata").read_text()
                )
        try:
            colbert_config_path = hf_hub_download(
                repo_id=str(model_name_or_path), filename="artifact.metadata"
            )
            colbert_config = json.loads(Path(colbert_config_path).read_text())
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if colbert_config is None:
            raise ValueError("{model_name_or_path} is not a valid colbert checkpoint.")
        config = ColBERTConfig.from_pretrained(model_name_or_path)
        config.update(
            {
                "similarity_function": colbert_config["similarity"],
                "query_aggregation_function": "sum",
                "doc_aggregation_function": "max",
                "query_expansion": True,
                "query_length": colbert_config["query_maxlen"],
                "attend_to_query_expanded_tokens": colbert_config[
                    "attend_to_mask_tokens"
                ],
                "doc_expansion": False,
                "doc_length": 512,
                "attend_to_doc_expanded_tokens": False,
                "normalize": True,
                "add_marker_tokens": True,
                "embedding_dim": colbert_config["dim"],
                "mask_punctuation": colbert_config["mask_punctuation"],
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


FlashColBERTModel = FlashClassFactory(ColBERTModel)


class ColBERTModule(MVRModule):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: MVRConfig | ColBERTConfig | None = None,
        loss_function: LossFunction | None = None,
    ) -> None:
        if model_name_or_path is None:
            if config is None:
                raise ValueError(
                    "Either model_name_or_path or config must be provided."
                )
            if not isinstance(config, ColBERTConfig):
                raise ValueError(
                    "config initializing a new model pass a ColBERTConfig."
                )
            model = FlashColBERTModel(config)
        else:
            model = FlashColBERTModel.from_pretrained(model_name_or_path, config=config)
        super().__init__(model, loss_function)
