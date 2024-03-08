from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoModel

from .colbert import ColBERTConfig, ColBERTModel
from .flash.flash_model import FlashClassFactory
from .loss import LossFunction
from .mvr import MVRConfig, MVRModule, ScoringFunction


class XTRConfig(ColBERTConfig):
    model_type = "xtr"

    ADDED_ARGS = ColBERTConfig.ADDED_ARGS + ["token_retrieval_k"]

    def __init__(
        self,
        token_retrieval_k: int | None = None,
        mask_punctuation: bool = True,
        **kwargs
    ) -> None:
        super().__init__(mask_punctuation, **kwargs)
        self.token_retrieval_k = token_retrieval_k

    def to_mvr_dict(self) -> Dict[str, Any]:
        mvr_dict = super().to_mvr_dict()
        mvr_dict["token_retrieval_k"] = self.token_retrieval_k
        return mvr_dict


class XTRScoringFunction(ScoringFunction):
    def __init__(self, config: XTRConfig) -> None:
        super().__init__(config)
        self.xtr_token_retrieval_k = config.token_retrieval_k

    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        similarity = super().compute_similarity(query_embeddings, doc_embeddings, mask)

        if self.xtr_token_retrieval_k is not None:
            top_k_similarity = similarity.topk(self.xtr_token_retrieval_k, dim=1)
            cut_off_similarity = top_k_similarity.values[:, [-1]]
            similarity = similarity.masked_fill(similarity < cut_off_similarity, 0)
        return similarity


class XTRModel(ColBERTModel):
    config_class = XTRConfig

    def __init__(self, xtr_config: XTRConfig) -> None:
        super().__init__(xtr_config)
        self.scoring_function = XTRScoringFunction(xtr_config)


FlashXTRModel = FlashClassFactory(XTRModel)


class XTRModule(MVRModule):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: MVRConfig | XTRConfig | None = None,
        loss_function: LossFunction | None = None,
    ) -> None:
        if model_name_or_path is None:
            if config is None:
                raise ValueError(
                    "Either model_name_or_path or config must be provided."
                )
            if not isinstance(config, XTRConfig):
                raise ValueError("config initializing a new model pass a XTRConfig.")
            model = FlashXTRModel(config)
        else:
            model = FlashXTRModel.from_pretrained(model_name_or_path, config=config)
        super().__init__(model, loss_function)


AutoConfig.register("xtr", XTRConfig)
AutoModel.register(XTRConfig, XTRModel)
