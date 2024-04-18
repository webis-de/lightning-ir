from typing import Any, Dict, Literal, Sequence

import torch
from transformers import AutoConfig, AutoModel

from ..flash.flash_model import FlashClassFactory
from ..loss.loss import LossFunction
from .bi_encoder import BiEncoderConfig, ScoringFunction
from .colbert import ColBERTConfig, ColBERTModel
from .module import BiEncoderModule


class XTRConfig(ColBERTConfig):
    model_type = "xtr"

    ADDED_ARGS = ColBERTConfig.ADDED_ARGS + ["token_retrieval_k", "fill_strategy"]

    def __init__(
        self,
        token_retrieval_k: int | None = None,
        fill_strategy: Literal["zero", "min"] = "zero",
        normalization: Literal["Z"] | None = "Z",
        mask_punctuation: bool = True,
        **kwargs
    ) -> None:
        super().__init__(mask_punctuation, **kwargs)
        self.token_retrieval_k = token_retrieval_k
        self.fill_strategy = fill_strategy
        self.normalization = normalization

    def to_added_args_dict(self) -> Dict[str, Any]:
        mvr_dict = super().to_added_args_dict()
        mvr_dict["token_retrieval_k"] = self.token_retrieval_k
        return mvr_dict


class XTRScoringFunction(ScoringFunction):
    def __init__(self, config: XTRConfig) -> None:
        super().__init__(config)
        self.xtr_token_retrieval_k = config.token_retrieval_k
        self.fill_strategy = config.fill_strategy
        self.normalization = config.normalization

    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: torch.Tensor,
    ) -> torch.Tensor:
        similarity = super().compute_similarity(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
            num_docs,
        )

        if self.training and self.xtr_token_retrieval_k is not None:
            if not torch.all(num_docs == num_docs[0]):
                raise ValueError(
                    "XTR token retrieval does not support variable number of documents."
                )
            query_embeddings = query_embeddings[:: num_docs[0]]
            doc_embeddings = doc_embeddings.view(1, 1, -1, doc_embeddings.shape[-1])
            ib_similarity = super().compute_similarity(
                query_embeddings,
                doc_embeddings,
                query_scoring_mask[:: num_docs[0]],
                doc_scoring_mask.view(1, -1),
                num_docs,
            )
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
        if self.training and self.normalization == "Z":
            # Z-normalization
            mask = mask & (scores != 0)
        return super().aggregate(scores, mask, aggregation_function)


class XTRModel(ColBERTModel):
    config_class = XTRConfig

    def __init__(self, xtr_config: XTRConfig) -> None:
        super().__init__(xtr_config)
        self.scoring_function = XTRScoringFunction(xtr_config)


FlashXTRModel = FlashClassFactory(XTRModel)


class XTRModule(BiEncoderModule):
    config_class = XTRConfig

    def __init__(
        self,
        model: XTRModel | None = None,
        model_name_or_path: str | None = None,
        config: BiEncoderConfig | XTRConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, XTRConfig):
                    raise ValueError("To initialize a new model pass a XTRConfig.")
                model = FlashXTRModel(config)
            else:
                model = FlashXTRModel.from_pretrained(model_name_or_path, config=config)
        super().__init__(model, loss_functions, evaluation_metrics)


AutoConfig.register("xtr", XTRConfig)
AutoModel.register(XTRConfig, XTRModel)
