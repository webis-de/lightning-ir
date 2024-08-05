from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers.modeling_utils import load_state_dict

from ...base import LightningIRModelClassFactory
from ...bi_encoder.model import BiEncoderEmbedding, ScoringFunction
from ..col import ColModel
from .config import XTRConfig


class XTRScoringFunction(ScoringFunction):
    def __init__(self, config: XTRConfig) -> None:
        super().__init__(config)
        self.config: XTRConfig

    def compute_similarity(
        self, query_embeddings: BiEncoderEmbedding, doc_embeddings: BiEncoderEmbedding
    ) -> torch.Tensor:
        similarity = super().compute_similarity(query_embeddings, doc_embeddings)

        if self.training and self.xtr_token_retrieval_k is not None:
            pass
            # TODO implement simulated token retrieval

            # if not torch.all(num_docs == num_docs[0]):
            #     raise ValueError("XTR token retrieval does not support variable number of documents.")
            # query_embeddings = query_embeddings[:: num_docs[0]]
            # doc_embeddings = doc_embeddings.view(1, 1, -1, doc_embeddings.shape[-1])
            # ib_similarity = super().compute_similarity(
            #     query_embeddings,
            #     doc_embeddings,
            #     query_scoring_mask[:: num_docs[0]],
            #     doc_scoring_mask.view(1, -1),
            #     num_docs,
            # )
            # top_k_similarity = ib_similarity.topk(self.xtr_token_retrieval_k, dim=-1)
            # cut_off_similarity = top_k_similarity.values[..., [-1]].repeat_interleave(num_docs, dim=0)
            # if self.fill_strategy == "min":
            #     fill = cut_off_similarity.expand_as(similarity)[similarity < cut_off_similarity]
            # elif self.fill_strategy == "zero":
            #     fill = 0
            # similarity[similarity < cut_off_similarity] = fill
        return similarity

    # def aggregate(
    #     self,
    #     scores: torch.Tensor,
    #     mask: torch.Tensor,
    #     query_aggregation_function: Literal["max", "sum", "mean", "harmonic_mean"],
    # ) -> torch.Tensor:
    #     if self.training and self.normalization == "Z":
    #         # Z-normalization
    #         mask = mask & (scores != 0)
    #     return super().aggregate(scores, mask, query_aggregation_function)


class XTRModel(ColModel):
    config_class = XTRConfig

    def __init__(self, config: XTRConfig, *args, **kwargs) -> None:
        super().__init__(config)
        self.scoring_function = XTRScoringFunction(config)
        self.config: XTRConfig

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path, *args, **kwargs) -> "XTRModel":
        try:
            hf_hub_download(repo_id=str(model_name_or_path), filename="2_Dense/pytorch_model.bin")
        except Exception:
            return super().from_pretrained(model_name_or_path, *args, **kwargs)
        finally:
            return cls.from_xtr_checkpoint(model_name_or_path)

    @classmethod
    def from_xtr_checkpoint(cls, model_name_or_path: Path | str) -> "XTRModel":
        from transformers import T5EncoderModel

        cls = LightningIRModelClassFactory(XTRConfig).from_backbone_class(T5EncoderModel)
        config = cls.config_class.from_pretrained(model_name_or_path)
        config.update(
            {
                "name_or_path": str(model_name_or_path),
                "similarity_function": "dot",
                "query_aggregation_function": "sum",
                "query_expansion": False,
                "doc_expansion": False,
                "doc_pooling_strategy": None,
                "doc_mask_scoring_tokens": None,
                "normalize": True,
                "sparsification": None,
                "add_marker_tokens": False,
                "embedding_dim": 128,
                "projection": "linear_no_bias",
            }
        )
        state_dict_path = hf_hub_download(repo_id=str(model_name_or_path), filename="model.safetensors")
        state_dict = load_state_dict(state_dict_path)
        linear_state_dict_path = hf_hub_download(repo_id=str(model_name_or_path), filename="2_Dense/pytorch_model.bin")
        linear_state_dict = load_state_dict(linear_state_dict_path)
        linear_state_dict["projection.weight"] = linear_state_dict.pop("linear.weight")
        state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
        state_dict.update(linear_state_dict)
        model = cls(config=config)
        model.load_state_dict(state_dict)
        return model
