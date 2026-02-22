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

    L2 Regularization, also known as Ridge Regression, adds a penalty term to the loss function that is proportional to
    the square of the magnitude of the model's parameters (in this case, the query and document embeddings). This
    encourages the model to keep the embeddings small, which can help prevent overfitting by discouraging the model
    from relying too heavily on any single feature. The L2 penalty is differentiable and leads to a smooth optimization
    landscape, making it a popular choice for regularization in machine learning models.

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

    L1 Regularization, also known as Lasso Regression, adds a penalty term to the loss function that is proportional to
    the absolute value of the model's parameters (in this case, the query and document embeddings). This encourages
    sparsity in the embeddings, meaning that it pushes many of the embedding dimensions to be exactly zero. This can
    lead to more interpretable models and can also help with feature selection by effectively removing less important
    features.

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

    FLOPS Regularization adds a penalty to the loss function that encourages the model to produce sparse embeddings,
    which can lead to more efficient inference by reducing the number of non-zero parameters. This is particularly
    beneficial for large-scale retrieval systems where computational efficiency is crucial. The FLOPS regularization
    term is designed to minimize the number of floating-point operations (FLOPS) required during inference by promoting
    sparsity in the embeddings, effectively encouraging the model to focus on the most important features while
    ignoring less relevant ones.

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
