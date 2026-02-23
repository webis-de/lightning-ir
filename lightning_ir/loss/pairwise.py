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

    MarginMSE optimizes pairwise ranking by penalizing the squared difference between the predicted score margin of a
    positive and negative document and a target margin. This target margin can be a fixed constant or dynamically
    derived from the difference in ground truth or teacher scores, making it particularly effective for knowledge
    distillation tasks where a student model learns to replicate the score distances of a stronger teacher model.

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

    RankNet optimizes pairwise ranking by modeling the probability that a positive document should be ranked higher
    than a negative document using a logistic function. It computes the margin between the scores of positive and
    negative pairs and applies a binary cross-entropy loss to maximize the likelihood of correct pairwise orderings.
    This approach allows the model to learn from relative comparisons rather than absolute score values.

    Originally proposed in: `Learning to Rank using Gradient Descent \
    <https://dl.acm.org/doi/10.1145/1102351.1102363>`_
    """

    def __init__(self, temperature: float = 1) -> None:
        """Initialize the RankNet loss function.
        Args:
            temperature (float): Temperature parameter for scaling the scores.
        """
        super().__init__()

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
