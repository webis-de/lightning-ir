"""
Listwise loss functions for the Lightning IR framework.

This module contains loss functions that operate on entire lists of items,
considering the ranking of all items simultaneously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import ListwiseLossFunction

if TYPE_CHECKING:
    from ..base import LightningIROutput
    from ..data import TrainBatch


class KLDivergence(ListwiseLossFunction):
    """Kullback-Leibler Divergence loss for listwise ranking tasks.

    KL Divergence loss for listwise ranking treats both the ground truth relevance labels and the predicted scores as
    probability distributions over the entire list of items. The loss is computed by minimizing the divergence between
    them to align the global ranking structure rather than just local comparisons.

    Originally proposed in: `On Information and Sufficiency \
    <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full>`_
    """

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the Kullback-Leibler Divergence loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        targets = torch.nn.functional.log_softmax(targets.to(scores), dim=-1)
        loss = torch.nn.functional.kl_div(scores, targets, log_target=True, reduction="batchmean")
        return loss


class PearsonCorrelation(ListwiseLossFunction):
    """Pearson Correlation loss for listwise ranking tasks.

    Pearson Correlation for listwise ranking maximizes the linear alignment between the vector of predicted scores and
    the vector of ground truth relevance labels, ensuring that the relative trends across the entire list are preserved
    regardless of the absolute scale of the scores.
    """

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the Pearson Correlation loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch).to(scores)
        centered_scores = scores - scores.mean(dim=-1, keepdim=True)
        centered_targets = targets - targets.mean(dim=-1, keepdim=True)
        pearson = torch.nn.functional.cosine_similarity(centered_scores, centered_targets, dim=-1)
        loss = (1 - pearson).mean()
        return loss


class InfoNCE(ListwiseLossFunction):
    """InfoNCE loss for listwise ranking tasks.

    Information Noise-Contrastive Estimation loss for listwise ranking adapts contrastive learning by treating the
    relevant item as the positive signal and all other items in the list as negative noise, maximizing the likelihood
    of the correct document relative to the entire candidate set via a softmax-normalized objective.

    Originally proposed in: `Representation Learning with Contrastive Predictive Coding \
    <https://arxiv.org/abs/1807.03748>`_
    """

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the InfoNCE loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        targets = targets.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss
