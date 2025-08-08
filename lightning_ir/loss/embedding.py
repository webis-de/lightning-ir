from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import EmbeddingLossFunction

if TYPE_CHECKING:
    from ..bi_encoder import BiEncoderOutput


class ContrastiveLocalLoss(EmbeddingLossFunction):
    """Loss function that computes a contrastive loss between a query and multiple document embeddings, such that only
    one document embedding has a a high similarity to the query embedding, while all other document embeddings
    have a low similarity. Originally proposed in:
    `Multi-View Document Representation Learning for Open-Domain Dense Retrieval \
    <https://aclanthology.org/2022.acl-long.414/>`_"""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the loss based on the embeddings in the output.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        similarity = output.similarity
        if similarity is None:
            raise ValueError("Expected similarity in BiEncoderOutput")
        targets = similarity.argmax(-1)
        loss = torch.nn.functional.cross_entropy(similarity, targets)
        return loss
