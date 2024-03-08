from typing import Any, Dict, List, Literal, Sequence, Tuple

import torch
from transformers import PretrainedConfig, PreTrainedModel


class MVRConfig(PretrainedConfig):
    model_type = "mvr"

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
        aggregation_function: Literal["sum", "mean", "max"] = "sum",
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

    def to_mvr_dict(self) -> Dict[str, Any]:
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
    ) -> "MVRConfig":
        return cls.from_dict({**config.to_dict(), **kwargs})


def ceil_div(a: int, b: int) -> int:
    return -(a // -b)


class ScoringFunction:

    def __init__(
        self,
        config: MVRConfig,
    ) -> None:
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
        mask: torch.Tensor,
    ) -> torch.Tensor:
        to_cpu = query_embeddings.is_cpu or doc_embeddings.is_cpu
        if torch.cuda.is_available():
            query_embeddings = query_embeddings.cuda()
            doc_embeddings = doc_embeddings.cuda()

        similarity = self.similarity_function(query_embeddings, doc_embeddings)
        if to_cpu:
            similarity = similarity.cpu()
        return similarity

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)

    def l2_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -1 * torch.cdist(x, y).squeeze(-2)

    def dot_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.transpose(-1, -2)).squeeze(-2)

    @staticmethod
    def aggregate(
        scores: torch.Tensor,
        mask: torch.Tensor,
        aggregation_function: Literal["max", "sum", "mean"],
    ) -> torch.Tensor:
        scores[~mask] = 0
        if aggregation_function == "max":
            return scores.max(-1).values
        if aggregation_function == "sum":
            return scores.sum(-1)
        if aggregation_function == "mean":
            num_non_masked = mask.sum(-1)
            return scores.sum(-1) / num_non_masked
        raise ValueError(f"Unknown aggregation {aggregation_function}")

    def _parse_num_docs(
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
                return torch.ones(1, 1, 1, dtype=torch.bool)
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

    def score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: int | List[int] | None = None,
    ) -> torch.Tensor:
        num_docs_t = self._parse_num_docs(query_embeddings, doc_embeddings, num_docs)

        exp_query_embeddings = query_embeddings.repeat_interleave(
            num_docs_t, dim=0
        ).unsqueeze(2)
        exp_doc_embeddings = doc_embeddings.unsqueeze(1)
        exp_query_scoring_mask = (
            query_scoring_mask.bool().repeat_interleave(num_docs_t, dim=0).unsqueeze(2)
        )
        exp_doc_scoring_mask = doc_scoring_mask.bool().unsqueeze(1)
        mask = exp_query_scoring_mask & exp_doc_scoring_mask

        similarity = self.compute_similarity(
            exp_query_embeddings, exp_doc_embeddings, mask
        )
        scores = self.aggregate(similarity, mask, "max")
        scores = self.aggregate(scores, mask.any(-1), self.aggregation_function)
        return scores


class MVRModel(PreTrainedModel):
    def __init__(self, config: MVRConfig, encoder: PreTrainedModel):
        super().__init__(config)
        self.config: MVRConfig
        self.encoder = encoder
        self.linear = torch.nn.Linear(
            self.config.hidden_size,
            self.config.embedding_dim,
            bias=self.config.linear_bias,
        )
        self.config.similarity_function
        self.scoring_function = ScoringFunction(config)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        doc_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
        query_token_type_ids: torch.Tensor | None = None,
        doc_token_type_ids: torch.Tensor | None = None,
        num_docs: List[int] | int | None = None,
    ) -> torch.Tensor:
        query_embeddings = self.encode(
            query_input_ids, query_attention_mask, query_token_type_ids
        )
        doc_embeddings = self.encode(
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
        return scores

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
