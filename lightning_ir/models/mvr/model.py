from dataclasses import dataclass
from typing import Literal, Sequence

from transformers import BatchEncoding

import torch

from lightning_ir.models.mvr.config import MVRConfig
from ...bi_encoder import BiEncoderModel, BiEncoderOutput


@dataclass
class MVROutput(BiEncoderOutput):
    """Dataclass containing the output of a MVR model."""

    viewer_token_scores: torch.tensor = None
    """individual similarity scores for each viewer token with query"""

class MVRModel(BiEncoderModel):
    config_class = MVRConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)


    def forward(
        self,
        query_encoding: BatchEncoding | None,
        doc_encoding: BatchEncoding | None,
        num_docs: Sequence[int] | int | None = None,
    ) -> MVROutput:

        query_embeddings = None
        if query_encoding is not None:
            query_embeddings = super().encode_query(query_encoding)
        doc_embeddings = None
        if doc_encoding is not None:
            doc_embeddings = super().encode_doc(doc_encoding)
        scores = None
        if doc_embeddings is not None and query_embeddings is not None:
            scores = self.score(query_embeddings, doc_embeddings, num_docs)
        return MVROutput(scores=scores[0], query_embeddings=query_embeddings, doc_embeddings=doc_embeddings, 
                         viewer_token_scores=scores[1])

