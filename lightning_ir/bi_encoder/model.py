from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple

import torch
from transformers import PretrainedConfig

from ..model import LightningIRConfig, LightningIRModel, LightningIROutput
from ..tokenizer.tokenizer import BiEncoderTokenizer

# TODO add configs for MultiVector and SingleVector models


class BiEncoderConfig(LightningIRConfig):
    model_type = "bi-encoder"
    Tokenizer = BiEncoderTokenizer

    ADDED_ARGS = [
        "similarity_function",
        "aggregation_function",
        "query_expansion",
        "query_length",
        "attend_to_query_expanded_tokens",
        "doc_expansion",
        "doc_length",
        "attend_to_doc_expanded_tokens",
        "normalize",
        "add_marker_tokens",
        "embedding_dim",
        "linear_bias",
    ]

    TOKENIZER_ARGS = [
        "query_expansion",
        "query_length",
        "attend_to_query_expanded_tokens",
        "doc_expansion",
        "doc_length",
        "attend_to_doc_expanded_tokens",
        "add_marker_tokens",
    ]

    def __init__(
        self,
        similarity_function: Literal["cosine", "l2", "dot"] = "dot",
        aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "sum",
        query_expansion: bool = False,
        query_length: int = 32,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        doc_length: int = 512,
        attend_to_doc_expanded_tokens: bool = False,
        normalize: bool = True,
        add_marker_tokens: bool = True,
        embedding_dim: int = 128,
        linear_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.similarity_function = similarity_function
        self.aggregation_function = aggregation_function
        self.query_expansion = query_expansion
        self.query_length = query_length
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.doc_length = doc_length
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens
        self.normalize = normalize
        self.add_marker_tokens = add_marker_tokens
        self.embedding_dim = embedding_dim
        self.linear_bias = linear_bias

    def to_added_args_dict(self) -> Dict[str, Any]:
        return {
            arg: getattr(self, arg) for arg in self.ADDED_ARGS if hasattr(self, arg)
        }

    def to_tokenizer_dict(self) -> Dict[str, Any]:
        return {arg: getattr(self, arg) for arg in self.TOKENIZER_ARGS}

    @classmethod
    def from_other(
        cls,
        config: PretrainedConfig,
        **kwargs,
    ) -> "BiEncoderConfig":
        return cls.from_dict({**config.to_dict(), **kwargs})


@dataclass
class BiEncoderOutput(LightningIROutput):
    query_embeddings: torch.Tensor | None = None
    doc_embeddings: torch.Tensor | None = None


class ScoringFunction(torch.nn.Module):
    def __init__(self, config: BiEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.similarity_function == "cosine":
            self.similarity_function = self.cosine_similarity
        elif self.config.similarity_function == "l2":
            self.similarity_function = self.l2_similarity
        elif self.config.similarity_function == "dot":
            self.similarity_function = self.dot_similarity
        else:
            raise ValueError(
                f"Unknown similarity function {self.config.similarity_function}"
            )
        self.aggregation_function = self.config.aggregation_function

    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: torch.Tensor,
    ) -> torch.Tensor:
        if torch.cuda.is_available():
            # bfloat16 similarity yields weird values with gpu, so we use fp16 instead
            # TODO investigate why, all values are a factor of 1/4
            query_embeddings = query_embeddings.cuda().half()
            doc_embeddings = doc_embeddings.cuda().half()

        similarity = self.similarity_function(query_embeddings, doc_embeddings)
        similarity = similarity.to(query_embeddings)
        similarity = similarity.masked_fill(~query_scoring_mask.unsqueeze(2), 0)
        similarity = similarity.masked_fill(~doc_scoring_mask.unsqueeze(1), 0)

        return similarity

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)

    def l2_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -1 * torch.cdist(x, y).squeeze(-2)

    def dot_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.transpose(-1, -2)).squeeze(-2)

    def aggregate(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        aggregation_function: Literal["max", "sum", "mean", "harmonic_mean"],
    ) -> torch.Tensor:
        if aggregation_function == "max":
            return scores.max(-1).values
        if aggregation_function == "sum":
            return scores.sum(-1)
        num_non_masked = mask.sum(-1)
        if aggregation_function == "mean":
            return torch.where(num_non_masked == 0, 0, scores.sum(-1) / num_non_masked)
        if aggregation_function == "harmonic_mean":
            return torch.where(
                num_non_masked == 0, 0, num_non_masked / (1 / scores).sum(-1)
            )
        raise ValueError(f"Unknown aggregation {aggregation_function}")

    def parse_num_docs(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        num_docs: int | Sequence[int] | None,
    ) -> torch.Tensor:
        batch_size = query_embeddings.shape[0]
        if isinstance(num_docs, int):
            num_docs = [num_docs] * batch_size
        if isinstance(num_docs, list):
            if sum(num_docs) != doc_embeddings.shape[0] or len(num_docs) != batch_size:
                raise ValueError("Num docs does not match doc embeddings")
        if num_docs is None:
            if doc_embeddings.shape[0] % batch_size != 0:
                raise ValueError(
                    "Docs are not evenly distributed in batch, but no num_docs provided"
                )
            num_docs = [doc_embeddings.shape[0] // batch_size] * batch_size
        return torch.tensor(num_docs, device=query_embeddings.device)

    def query_scoring_mask(
        self,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scoring_mask(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            query_expansion=self.config.query_expansion,
        )

    def doc_scoring_mask(
        self,
        doc_input_ids: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scoring_mask(
            doc_input_ids, doc_attention_mask, self.config.doc_expansion
        )

    def scoring_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        query_expansion: bool,
    ) -> torch.Tensor:
        if input_ids is None:
            if attention_mask is None:
                raise ValueError("Pass either input_ids or attention_mask")
            else:
                shape = attention_mask.shape
                device = attention_mask.device
        else:
            shape = input_ids.shape
            device = input_ids.device
        scoring_mask = attention_mask
        if query_expansion or scoring_mask is None:
            scoring_mask = torch.ones(shape, dtype=torch.bool, device=device)
        scoring_mask = scoring_mask.bool()
        return scoring_mask

    def scoring_masks(
        self,
        query_input_ids: torch.Tensor | None = None,
        doc_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_scoring_mask = self.query_scoring_mask(
            query_input_ids, query_attention_mask
        )
        doc_scoring_mask = self.doc_scoring_mask(doc_input_ids, doc_attention_mask)
        return query_scoring_mask, doc_scoring_mask

    def expand_query_embeddings(
        self,
        query_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        num_docs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_embeddings = query_embeddings.repeat_interleave(
            num_docs, dim=0
        ).unsqueeze(2)
        if query_scoring_mask.numel() > 1:
            query_scoring_mask = query_scoring_mask.repeat_interleave(num_docs, dim=0)
        return query_embeddings, query_scoring_mask

    def expand_doc_embeddings(
        self,
        doc_embeddings: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        doc_embeddings = doc_embeddings.unsqueeze(1)
        return doc_embeddings, doc_scoring_mask

    def score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: int | List[int] | None = None,
    ) -> torch.Tensor:
        num_docs_t = self.parse_num_docs(query_embeddings, doc_embeddings, num_docs)
        query_embeddings, query_scoring_mask = self.expand_query_embeddings(
            query_embeddings, query_scoring_mask, num_docs_t
        )
        doc_embeddings, doc_scoring_mask = self.expand_doc_embeddings(
            doc_embeddings, doc_scoring_mask, num_docs_t
        )

        similarity = self.compute_similarity(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
            num_docs_t,
        )
        scores = self.aggregate(similarity, doc_scoring_mask, "max")
        scores = self.aggregate(scores, query_scoring_mask, self.aggregation_function)
        return scores


class BiEncoderModel(LightningIRModel):
    def __init__(self, config: BiEncoderConfig, encoder_module_name: str):
        super().__init__(config)
        self.config: BiEncoderConfig
        self.encoder_module_name = encoder_module_name
        self.linear = torch.nn.Linear(
            self.config.hidden_size,
            self.config.embedding_dim,
            bias=self.config.linear_bias,
        )
        self.scoring_function = ScoringFunction(config)

    @property
    def encoder(self) -> torch.nn.Module:
        return getattr(self, self.encoder_module_name)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        doc_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
        query_token_type_ids: torch.Tensor | None = None,
        doc_token_type_ids: torch.Tensor | None = None,
        num_docs: List[int] | int | None = None,
    ) -> BiEncoderOutput:
        query_embeddings = self.encode_queries(
            query_input_ids, query_attention_mask, query_token_type_ids
        )
        doc_embeddings = self.encode_docs(
            doc_input_ids, doc_attention_mask, doc_token_type_ids
        )
        query_scoring_mask, doc_scoring_mask = self.scoring_masks(
            query_input_ids, doc_input_ids, query_attention_mask, doc_attention_mask
        )
        scores = self.score(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
            num_docs,
        )
        return BiEncoderOutput(
            scores=scores,
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.encoder.forward(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state
        embedding = self.linear(embedding)
        if self.config.normalize:
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(input_ids, attention_mask, token_type_ids)

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(input_ids, attention_mask, token_type_ids)

    def scoring_masks(
        self,
        query_input_ids: torch.Tensor | None = None,
        doc_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.scoring_function.scoring_masks(
            query_input_ids, doc_input_ids, query_attention_mask, doc_attention_mask
        )

    def score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: List[int] | int | None = None,
    ) -> torch.Tensor:
        if (
            query_scoring_mask.dtype != torch.bool
            or doc_scoring_mask.dtype != torch.bool
        ):
            raise ValueError("Scoring masks must be boolean")
        scores = self.scoring_function.score(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask=query_scoring_mask,
            doc_scoring_mask=doc_scoring_mask,
            num_docs=num_docs,
        )
        return scores
