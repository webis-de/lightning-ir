from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from transformers import BatchEncoding

from ..base import LightningIRModel, LightningIROutput
from . import BiEncoderConfig, MultiVectorBiEncoderConfig, SingleVectorBiEncoderConfig


@dataclass
class BiEncoderEmbedding:
    embeddings: torch.Tensor
    scoring_mask: torch.Tensor


@dataclass
class BiEncoderOutput(LightningIROutput):
    query_embeddings: BiEncoderEmbedding | None = None
    doc_embeddings: BiEncoderEmbedding | None = None


class BiEncoderModel(LightningIRModel):
    config_class = BiEncoderConfig

    def __init__(self, config: BiEncoderConfig) -> None:
        super().__init__(config)
        self.config: BiEncoderConfig
        self.scoring_function = ScoringFunction(self.config)
        self.linear = None
        if self.config.linear:
            self.linear = torch.nn.Linear(
                self.config.hidden_size,
                self.config.embedding_dim,
                bias=self.config.linear_bias,
            )
        else:
            if self.config.embedding_dim != self.config.hidden_size:
                raise ValueError(
                    "Embedding dim must match hidden size if no linear layer is used"
                )

    def forward(
        self,
        query_encoding: BatchEncoding,
        doc_encoding: BatchEncoding,
        num_docs: Sequence[int] | int | None = None,
    ) -> BiEncoderOutput:
        query_embeddings = self.encode_query(**query_encoding)
        doc_embeddings = self.encode_doc(**doc_encoding)
        scores = self.score(
            query_embeddings,
            doc_embeddings,
            num_docs,
        )
        return BiEncoderOutput(
            scores=scores,
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
        )

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> BiEncoderEmbedding:
        return self._encode(
            input_ids, attention_mask, token_type_ids, self.config.query_expansion
        )

    def encode_doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> BiEncoderEmbedding:
        return self._encode(
            input_ids, attention_mask, token_type_ids, self.config.doc_expansion
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        expansion: bool = False,
    ) -> BiEncoderEmbedding:
        embeddings = self.backbone_forward(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state
        embeddings = self.process_embeddings(embeddings)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self._scoring_mask(input_ids, attention_mask, expansion)
        return BiEncoderEmbedding(embeddings, scoring_mask)

    def process_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.linear is not None:
            embeddings = self.linear(embeddings)
        return embeddings

    def query_scoring_mask(
        self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return self._scoring_mask(
            input_ids, attention_mask, self.config.query_expansion
        )

    def doc_scoring_mask(
        self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return self._scoring_mask(
            input_ids, attention_mask, self.config.query_expansion
        )

    def _scoring_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        expansion: bool,
    ) -> torch.Tensor:
        raise NotImplementedError("Scoring mask method must be implemented in subclass")

    def score(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        scores = self.scoring_function.score(
            query_embeddings, doc_embeddings, num_docs=num_docs
        )
        return scores


class SingleVectorBiEncoderModel(BiEncoderModel):
    config_class = SingleVectorBiEncoderConfig

    def __init__(self, config: SingleVectorBiEncoderConfig) -> None:
        super().__init__(config)
        self.config: SingleVectorBiEncoderConfig
        if self.config.pooling_strategy is None:
            raise ValueError("Pooling strategy must be set")

    def process_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        embeddings = super().process_embeddings(embeddings, attention_mask)
        embeddings = self.pooling(embeddings, attention_mask).unsqueeze(1)
        return embeddings

    def _scoring_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        expansion: bool,
    ) -> torch.Tensor:
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
        elif attention_mask is not None:
            batch_size = attention_mask.shape[0]
            device = attention_mask.device
        else:
            raise ValueError("Pass either input_ids or attention_mask")
        return torch.ones((batch_size, 1), dtype=torch.bool, device=device)


class MultiVectorBiEncoderModel(BiEncoderModel):
    config_class = MultiVectorBiEncoderConfig

    def __init__(self, config: MultiVectorBiEncoderConfig) -> None:
        super().__init__(config)
        self.config: MultiVectorBiEncoderConfig

    def _scoring_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        expansion: bool,
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
        if expansion or scoring_mask is None:
            scoring_mask = torch.ones(shape, dtype=torch.bool, device=device)
        scoring_mask = scoring_mask.bool()
        return scoring_mask


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
        self.aggregation_function = getattr(self.config, "aggregation_function", None)

    def compute_similarity(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
    ) -> torch.Tensor:
        # if torch.cuda.is_available():
        #     # bfloat16 similarity yields weird values with gpu, so we use fp16 instead
        #     # TODO investigate why, all values are a factor of 1/4
        #     query_tensor = query_tensor.cuda().half()
        #     doc_tensor = doc_tensor.cuda().half()

        # TODO compute similarity only for non-masked values
        similarity = self.similarity_function(
            query_embeddings.embeddings, doc_embeddings.embeddings
        )
        return similarity

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)

    def l2_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -1 * torch.cdist(x, y).squeeze(-2)

    def dot_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.transpose(-1, -2)).squeeze(-2)

    def parse_num_docs(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: int | Sequence[int] | None,
    ) -> torch.Tensor:
        batch_size = query_embeddings.embeddings.shape[0]
        if isinstance(num_docs, int):
            num_docs = [num_docs] * batch_size
        if isinstance(num_docs, list):
            if (
                sum(num_docs) != doc_embeddings.embeddings.shape[0]
                or len(num_docs) != batch_size
            ):
                raise ValueError("Num docs does not match doc embeddings")
        if num_docs is None:
            if doc_embeddings.embeddings.shape[0] % batch_size != 0:
                raise ValueError(
                    "Docs are not evenly distributed in batch, but no num_docs provided"
                )
            num_docs = [doc_embeddings.embeddings.shape[0] // batch_size] * batch_size
        return torch.tensor(num_docs, device=query_embeddings.embeddings.device)

    def expand_query_embeddings(
        self,
        embeddings: BiEncoderEmbedding,
        num_docs: torch.Tensor,
    ) -> BiEncoderEmbedding:
        return BiEncoderEmbedding(
            embeddings.embeddings.repeat_interleave(num_docs, dim=0).unsqueeze(2),
            embeddings.scoring_mask.repeat_interleave(num_docs, dim=0).unsqueeze(2),
        )

    def expand_doc_embeddings(
        self,
        embeddings: BiEncoderEmbedding,
        num_docs: torch.Tensor,
    ) -> BiEncoderEmbedding:
        return BiEncoderEmbedding(
            embeddings.embeddings.unsqueeze(1), embeddings.scoring_mask.unsqueeze(1)
        )

    def aggregate(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor | None,
        aggregation_function: Literal["max", "sum", "mean", "harmonic_mean"] | None,
    ) -> torch.Tensor:
        if aggregation_function is None:
            return scores
        if aggregation_function == "max":
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            return scores.max(-1, keepdim=True).values
        if aggregation_function == "sum":
            if mask is not None:
                scores = scores.masked_fill(~mask, 0)
            return scores.sum(-1, keepdim=True)
        if mask is None:
            num_non_masked = torch.full(
                scores.shape[:-1], scores.shape[-1], device=scores.device
            )
        else:
            num_non_masked = mask.sum(-1)
        if aggregation_function == "mean":
            return torch.where(
                num_non_masked == 0, 0, scores.sum(-1, keepdim=True) / num_non_masked
            )
        if aggregation_function == "harmonic_mean":
            return torch.where(
                num_non_masked == 0,
                0,
                num_non_masked / (1 / scores).sum(-1, keepdim=True),
            )
        raise ValueError(f"Unknown aggregation {aggregation_function}")

    def score(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        num_docs_t = self.parse_num_docs(query_embeddings, doc_embeddings, num_docs)
        query_embeddings = self.expand_query_embeddings(query_embeddings, num_docs_t)
        doc_embeddings = self.expand_doc_embeddings(doc_embeddings, num_docs_t)
        similarity = self.compute_similarity(query_embeddings, doc_embeddings)
        scores = self.aggregate(similarity, doc_embeddings.scoring_mask, "max")
        scores = self.aggregate(
            scores, query_embeddings.scoring_mask, self.aggregation_function
        )
        return scores[..., 0, 0]
