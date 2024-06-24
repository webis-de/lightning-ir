from dataclasses import dataclass
from pathlib import Path
from string import punctuation
from typing import Literal, Sequence

import torch
from transformers import BatchEncoding

from ..base import LightningIRModel, LightningIROutput
from . import BiEncoderConfig


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

        self.query_mask_scoring_input_ids: torch.Tensor | None = None
        self.doc_mask_scoring_input_ids: torch.Tensor | None = None
        for sequence in ("query", "doc"):
            mask_scoring_tokens = getattr(
                self.config, f"{sequence}_mask_scoring_tokens"
            )
            if mask_scoring_tokens is None:
                continue
            if mask_scoring_tokens == "punctuation":
                mask_scoring_tokens = list(punctuation)
            try:
                tokenizer = self.config.__class__.tokenizer_class.from_pretrained(
                    self.config.name_or_path
                )
            except OSError:
                raise ValueError(
                    "Can't use token scoring masking if the checkpoint does not "
                    "have a tokenizer."
                )
            setattr(
                self,
                f"{sequence}_mask_scoring_input_ids",
                tokenizer(
                    mask_scoring_tokens,
                    add_special_tokens=False,
                    return_tensors="pt",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    warn=False,
                ).input_ids[:, 0],
            )

    def forward(
        self,
        query_encoding: BatchEncoding | None,
        doc_encoding: BatchEncoding | None,
        num_docs: Sequence[int] | int | None = None,
    ) -> BiEncoderOutput:
        query_embeddings = None
        if query_encoding is not None:
            query_embeddings = self.encode_query(**query_encoding)
        doc_embeddings = None
        if doc_encoding is not None:
            doc_embeddings = self.encode_doc(**doc_encoding)
        scores = None
        if doc_embeddings is not None and query_embeddings is not None:
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
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            expansion=self.config.query_expansion,
            pooling_strategy=self.config.query_pooling_strategy,
            mask_scoring_input_ids=self.query_mask_scoring_input_ids,
        )

    def encode_doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> BiEncoderEmbedding:
        return self._encode(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            expansion=self.config.doc_expansion,
            pooling_strategy=self.config.doc_pooling_strategy,
            mask_scoring_input_ids=self.doc_mask_scoring_input_ids,
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        expansion: bool = False,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
        mask_scoring_input_ids: torch.Tensor | None = None,
    ) -> BiEncoderEmbedding:
        embeddings = self.backbone_forward(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state
        if self.linear is not None:
            embeddings = self.linear(embeddings)
        embeddings = self.pooling(embeddings, attention_mask, pooling_strategy)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self._scoring_mask(
            input_ids,
            attention_mask,
            expansion,
            pooling_strategy,
            mask_scoring_input_ids,
        )
        return BiEncoderEmbedding(embeddings, scoring_mask)

    def query_scoring_mask(
        self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return self._scoring_mask(
            input_ids,
            attention_mask,
            expansion=self.config.query_expansion,
            pooling_strategy=self.config.query_pooling_strategy,
            mask_scoring_input_ids=self.config.query_mask_scoring_input_ids,
        )

    def doc_scoring_mask(
        self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return self._scoring_mask(
            input_ids,
            attention_mask,
            expansion=self.config.query_expansion,
            pooling_strategy=self.config.doc_pooling_strategy,
            mask_scoring_input_ids=self.config.doc_mask_scoring_input_ids,
        )

    def _scoring_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        expansion: bool,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
        mask_scoring_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            shape = input_ids.shape
            device = input_ids.device
        elif attention_mask is not None:
            shape = attention_mask.shape
            device = attention_mask.device
        else:
            raise ValueError("Pass either input_ids or attention_mask")
        if pooling_strategy is not None:
            return torch.ones((shape[0], 1), dtype=torch.bool, device=device)
        scoring_mask = attention_mask
        if expansion or scoring_mask is None:
            scoring_mask = torch.ones(shape, dtype=torch.bool, device=device)
        scoring_mask = scoring_mask.bool()
        if mask_scoring_input_ids is not None and input_ids is not None:
            ignore_mask = (
                input_ids[..., None].eq(mask_scoring_input_ids.to(device)).any(-1)
            )
            scoring_mask = scoring_mask & ~ignore_mask
        return scoring_mask

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
        self.doc_aggregation_function = self.config.doc_aggregation_function

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
        doc_aggregation_function: Literal["max", "sum", "mean", "harmonic_mean"] | None,
        dim: int,
    ) -> torch.Tensor:
        if doc_aggregation_function is None:
            return scores
        if doc_aggregation_function == "max":
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            return scores.max(dim, keepdim=True).values
        if doc_aggregation_function == "sum":
            if mask is not None:
                scores = scores.masked_fill(~mask, 0)
            return scores.sum(dim, keepdim=True)
        if mask is None:
            shape = list(scores.shape)
            shape[dim] = 1
            num_non_masked = torch.full(shape, scores.shape[dim], device=scores.device)
        else:
            num_non_masked = mask.sum(dim, keepdim=True)
        if doc_aggregation_function == "mean":
            return torch.where(
                num_non_masked == 0, 0, scores.sum(dim, keepdim=True) / num_non_masked
            )
        if doc_aggregation_function == "harmonic_mean":
            return torch.where(
                num_non_masked == 0,
                0,
                num_non_masked / (1 / scores).sum(dim, keepdim=True),
            )
        raise ValueError(f"Unknown aggregation {doc_aggregation_function}")

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
        scores = self.aggregate(similarity, doc_embeddings.scoring_mask, "max", -1)
        scores = self.aggregate(
            scores, query_embeddings.scoring_mask, self.doc_aggregation_function, -2
        )
        return scores[..., 0, 0]
