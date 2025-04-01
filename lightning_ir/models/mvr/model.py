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


    def scoring_mask(
        self,
        encoding: BatchEncoding,
        expansion: bool = False,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
        mask_scoring_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes a scoring for batched tokenized text sequences which is used in the scoring function to mask out
        vectors during scoring.

        :param encoding: Tokenizer encodings for the text sequence
        :type encoding: BatchEncoding
        :param expansion: Whether or not mask expansion was applied to the tokenized sequence, defaults to False
        :type expansion: bool, optional
        :param pooling_strategy: Which pooling strategy is pool the embeddings, defaults to None
        :type pooling_strategy: Literal['first', 'mean', 'max', 'sum'] | None, optional
        :param mask_scoring_input_ids: Sequence of token_ids which should be masked during scoring, defaults to None
        :type mask_scoring_input_ids: torch.Tensor | None, optional
        :return: Scoring mask
        :rtype: torch.Tensor
        """
        device = encoding["input_ids"].device
        input_ids: torch.Tensor = encoding["input_ids"]
        attention_mask: torch.Tensor = encoding["attention_mask"]
        shape = input_ids.shape
        if pooling_strategy is not None:
            if self.config.num_viewer_tokens is not None:
                return torch.ones((shape[0], (self.config.num_viewer_tokens)), dtype=torch.bool, device=device)
            return torch.ones((shape[0], 1), dtype=torch.bool, device=device)
        return super().scoring_mask