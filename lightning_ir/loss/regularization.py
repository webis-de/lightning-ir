"""
Regularization loss functions for the Lightning IR framework.

This module contains loss functions that apply regularization to embeddings
to prevent overfitting and improve generalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import RegularizationLossFunction

if TYPE_CHECKING:
    from ..bi_encoder import BiEncoderOutput


class L2Regularization(RegularizationLossFunction):
    """L2 Regularization loss function for query and document embeddings.
    Originally proposed in: `Ridge Regression: Biased Estimation for Nonorthogonal Problems
    <https://homepages.math.uic.edu/~lreyzin/papers/ridge.pdf>`_"""

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the L2 regularization loss.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        query_embeddings, doc_embeddings = self.process_embeddings(output)
        query_loss = self.query_weight * query_embeddings.norm(dim=-1).mean()
        doc_loss = self.doc_weight * doc_embeddings.norm(dim=-1).mean()
        loss = query_loss + doc_loss
        return loss


class L1Regularization(RegularizationLossFunction):
    """L1 Regularization loss function for query and document embeddings.
    Originally proposed in: `Regression Shrinkage and Selection via the Lasso
    <https://academic.oup.com/jrsssb/article/58/1/267/7027929>`_"""

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the L1 regularization loss.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        query_embeddings, doc_embeddings = self.process_embeddings(output)
        query_loss = self.query_weight * query_embeddings.norm(p=1, dim=-1).mean()
        doc_loss = self.doc_weight * doc_embeddings.norm(p=1, dim=-1).mean()
        loss = query_loss + doc_loss
        return loss


class FLOPSRegularization(RegularizationLossFunction):
    """FLOPS Regularization loss function for query and document embeddings.
    Originally proposed in: `Minimizing FLOPS to Learn Efficient Sparse Representations
    <https://arxiv.org/pdf/2004.05665>`_"""

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the FLOPS regularization loss.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        query_embeddings, doc_embeddings = self.process_embeddings(output)
        query_loss = torch.sum(torch.mean(torch.abs(query_embeddings), dim=0) ** 2)
        doc_loss = torch.sum(torch.mean(torch.abs(doc_embeddings), dim=0) ** 2)
        loss = self.query_weight * query_loss + self.doc_weight * doc_loss
        return loss
