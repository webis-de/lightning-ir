from typing import Any, Dict, Literal, Sequence

import torch
from transformers import AutoConfig, AutoModel

from .colbert import ColBERTConfig, ColBERTModel
from .flash.flash_model import FlashClassFactory
from .loss import LossFunction
from .module import MVRModule
from .mvr import MVRConfig, ScoringFunction


class XTRConfig(ColBERTConfig):
    model_type = "xtr"

    ADDED_ARGS = ColBERTConfig.ADDED_ARGS + ["token_retrieval_k", "fill_strategy"]

    def __init__(
        self,
        token_retrieval_k: int | None = None,
        fill_strategy: Literal["zero", "min"] = "zero",
        mask_punctuation: bool = True,
        **kwargs
    ) -> None:
        super().__init__(mask_punctuation, **kwargs)
        self.token_retrieval_k = token_retrieval_k
        self.fill_strategy = fill_strategy

    def to_mvr_dict(self) -> Dict[str, Any]:
        mvr_dict = super().to_mvr_dict()
        mvr_dict["token_retrieval_k"] = self.token_retrieval_k
        return mvr_dict


class XTRScoringFunction(ScoringFunction):
    def __init__(self, config: XTRConfig) -> None:
        super().__init__(config)
        self.xtr_token_retrieval_k = config.token_retrieval_k
        self.fill_strategy = config.fill_strategy

    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        mask: torch.Tensor,
        num_docs: torch.Tensor,
    ) -> torch.Tensor:
        similarity = super().compute_similarity(
            query_embeddings, doc_embeddings, mask, num_docs
        )

        if self.training and self.xtr_token_retrieval_k is not None:
            if not torch.all(num_docs == num_docs[0]):
                raise ValueError(
                    "XTR token retrieval does not support variable number of documents."
                )
            query_embeddings = query_embeddings[:: num_docs[0]]
            doc_embeddings = doc_embeddings.view(1, 1, -1, doc_embeddings.shape[-1])
            ib_similarity = super().compute_similarity(query_embeddings, doc_embeddings)
            top_k_similarity = ib_similarity.topk(self.xtr_token_retrieval_k, dim=-1)
            cut_off_similarity = top_k_similarity.values[..., [-1]].repeat_interleave(
                num_docs, dim=0
            )
            if self.fill_strategy == "min":
                fill = cut_off_similarity.expand_as(similarity)[
                    similarity < cut_off_similarity
                ]
            elif self.fill_strategy == "zero":
                fill = 0
            similarity[similarity < cut_off_similarity] = fill
        return similarity

    def aggregate(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        aggregation_function: Literal["max", "sum", "mean", "harmonic_mean"],
    ) -> torch.Tensor:
        # Z-normalization
        mask = mask & (scores != 0)
        return super().aggregate(scores, mask, aggregation_function)


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
        loss_functions: Sequence[LossFunction] | None = None,
    ) -> None:
        if model_name_or_path is None:
            if config is None:
                raise ValueError(
                    "Either model_name_or_path or config must be provided."
                )
            if not isinstance(config, XTRConfig):
                raise ValueError("To initialize a new model pass a XTRConfig.")
            model = FlashXTRModel(config)
        else:
            model = FlashXTRModel.from_pretrained(model_name_or_path, config=config)
        super().__init__(model, loss_functions)


AutoConfig.register("xtr", XTRConfig)
AutoModel.register(XTRConfig, XTRModel)
