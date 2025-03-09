
import warnings
from ..bi_encoder import (
    BiEncoderConfig,
    BiEncoderEmbedding,
    BiEncoderModel,
    ScoringFunction,
)

from typing import Literal, Sequence

from dataclasses import dataclass

from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import normalize

from transformers import BatchEncoding

@dataclass
class CoilEmbedding(BiEncoderEmbedding):
    cls_embeddings: Tensor

class CoilScoringFunction(ScoringFunction):

    def forward(
        self,
        query_embeddings: CoilEmbedding,
        doc_embeddings: CoilEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> Tensor:
        
        num_docs_t = self._parse_num_docs(query_embeddings, doc_embeddings, num_docs)
        query_cls_embeddings = (query_embeddings.cls_embeddings).repeat_interleave(num_docs_t, 0).unsqueeze(2)
        doc_cls_embeddings = doc_embeddings.cls_embeddings.unsqueeze(1)

        query_embeddings = self._expand_query_embeddings(query_embeddings, num_docs_t)
        doc_embeddings = self._expand_doc_embeddings(doc_embeddings, num_docs_t)
        similarity = self._compute_similarity(query_embeddings, doc_embeddings)

        query = query_embeddings.encoding.input_ids.repeat_interleave(num_docs_t, 0)[:, 1:]
        docs = doc_embeddings.encoding.input_ids[:, 1:]  

        mask = (query[:, :, None] == docs[:, None, :]).float()  

        cls_similarity = self.similarity_function(query_cls_embeddings, doc_cls_embeddings)

        scores = self._aggregate(mask * similarity, doc_embeddings.scoring_mask[:, :, 1:], "max", -1) 
        scores = self._aggregate(scores, query_embeddings.scoring_mask[:, 1:, :], self.query_aggregation_function, -2)
        return scores[..., 0, 0] + cls_similarity[..., 0, 0]


class CoilConfig(BiEncoderConfig):
    model_type = "coil"

    def __init__(
        self,
        query_expansion: bool = False,
        embedding_dim: int = 32,
        cls_embedding_dim: int = 768,
        projection: Literal["linear", "linear_no_bias"] | None = "linear_no_bias",
        **kwargs,
    ) -> None:
        kwargs["query_pooling_strategy"] = None
        kwargs["doc_expansion"] = False
        kwargs["attend_to_doc_expanded_tokens"] = False
        kwargs["doc_pooling_strategy"] = None
        super().__init__(
            query_expansion=query_expansion,
            embedding_dim=embedding_dim,
            projection=projection,
            **kwargs,
        )
        self.cls_embedding_dim = cls_embedding_dim


class CoilModel(BiEncoderModel):
    config_class = CoilConfig

    def __init__(self, config: BiEncoderConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.scoring_function = CoilScoringFunction(self.config)

        if self.config.projection is not None:
            if "linear" in self.config.projection:
                self.cls_projection = Linear(
                    self.config.hidden_size,
                    self.config.cls_embedding_dim,
                    bias="no_bias" not in self.config.projection,
                )
            else:
                raise ValueError(f"Projection {self.config.projection} for COIL is not supported")
        else:
            if self.config.embedding_dim != self.config.hidden_size:
                warnings.warn(
                    "No projection is used but embedding_dim != hidden_size. "
                    "The output embeddings will not have embedding_size dimensions."
                )
    
    
    def encode(
        self,
        encoding: BatchEncoding,
        expansion: bool = False,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
        mask_scoring_input_ids: Tensor | None = None,
    ) -> BiEncoderEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings and scoring mask.

        :param encoding: Tokenizer encodings for the text sequence
        :type encoding: BatchEncoding
        :param expansion: Whether mask expansion was applied to the text sequence, defaults to False
        :type expansion: bool, optional
        :param pooling_strategy: Strategy to pool token embeddings into a single embedding. If None no pooling is
            applied, defaults to None
        :type pooling_strategy: Literal['first', 'mean', 'max', 'sum'] | None, optional
        :param mask_scoring_input_ids: Which token_ids to mask out during scoring, defaults to None
        :type mask_scoring_input_ids: torch.Tensor | None, optional
        :return: Embeddings and scoring mask
        :rtype: BiEncoderEmbedding
        """
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        
        if self.projection is not None:
            cls_embeddings = self.cls_projection(embeddings[:, [0]])
            embeddings = self.projection(embeddings[:, 1:])

        embeddings = self._sparsification(embeddings, self.config.sparsification)
        embeddings = self._pooling(embeddings, encoding["attention_mask"], pooling_strategy)
        
        if self.config.normalize:
            embeddings = normalize(embeddings, dim=-1)

        scoring_mask = self.scoring_mask(encoding, expansion, pooling_strategy, mask_scoring_input_ids)
        return CoilEmbedding(embeddings, scoring_mask, encoding, cls_embeddings)
