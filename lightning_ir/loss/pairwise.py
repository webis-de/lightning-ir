"""
Pairwise loss functions for the Lightning IR framework.

This module contains loss functions that operate on pairs of items,
comparing positive and negative examples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .base import PairwiseLossFunction

if TYPE_CHECKING:
    from ..base import LightningIROutput
    from ..data import TrainBatch


class MarginMSE(PairwiseLossFunction):
    """Mean Squared Error loss with a margin for pairwise ranking tasks.
    Originally proposed in: `Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation \
    <https://arxiv.org/abs/2010.02666>`_
    """

    def __init__(self, margin: float | Literal["scores"] = 1.0):
        """Initialize the MarginMSE loss function.

        Args:
            margin (float | Literal["scores"]): The margin value for the loss.
        """
        self.margin = margin

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the MarginMSE loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        Raises:
            ValueError: If the margin type is invalid.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(targets)
        pos = scores[query_idcs, pos_idcs]
        neg = scores[query_idcs, neg_idcs]
        margin = pos - neg
        if isinstance(self.margin, float):
            target_margin = torch.tensor(self.margin, device=scores.device)
        elif self.margin == "scores":
            target_margin = targets[query_idcs, pos_idcs] - targets[query_idcs, neg_idcs]
        else:
            raise ValueError("invalid margin type")
        loss = torch.nn.functional.mse_loss(margin, target_margin)
        return loss


class ConstantMarginMSE(MarginMSE):
    """Constant Margin MSE loss for pairwise ranking tasks with a fixed margin."""

    def __init__(self, margin: float = 1.0):
        """Initialize the ConstantMarginMSE loss function.

        Args:
            margin (float): The fixed margin value for the loss.
        """
        super().__init__(margin)


class SupervisedMarginMSE(MarginMSE):
    """Supervised Margin MSE loss for pairwise ranking tasks with a dynamic margin."""

    def __init__(self):
        """Initialize the SupervisedMarginMSE loss function."""
        super().__init__("scores")


class RankNet(PairwiseLossFunction):
    """RankNet loss function for pairwise ranking tasks.
    Originally proposed in: `Learning to Rank using Gradient Descent \
    <https://dl.acm.org/doi/10.1145/1102351.1102363>`_
    """

    def __init__(self, temperature: float = 1) -> None:
        super().__init__()
        """Initialize the RankNet loss function.
        Args:
            temperature (float): Temperature parameter for scaling the scores.
        """
        self.temperature = temperature

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the RankNet loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output) * self.temperature
        targets = self.process_targets(scores, batch)
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(targets)
        pos = scores[query_idcs, pos_idcs]
        neg = scores[query_idcs, neg_idcs]
        margin = pos - neg
        loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        return loss
