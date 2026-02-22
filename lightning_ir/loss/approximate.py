"""
Approximate ranking loss functions for the Lightning IR framework.

This module contains loss functions that use approximation techniques to compute
ranking-based metrics like NDCG, MRR, and RankMSE in a differentiable manner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .base import ListwiseLossFunction

if TYPE_CHECKING:
    from ..base import LightningIROutput
    from ..data import TrainBatch


class ApproxLossFunction(ListwiseLossFunction):
    """Base class for approximate loss functions that compute ranks from scores."""

    def __init__(self, temperature: float = 1) -> None:
        """Initialize the ApproxLossFunction.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
        """
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def get_approx_ranks(scores: torch.Tensor, temperature: float) -> torch.Tensor:
        """Compute approximate ranks from scores.

        Args:
            scores (torch.Tensor): The input scores.
            temperature (float): Temperature parameter for scaling the scores.
        Returns:
            torch.Tensor: The computed approximate ranks.
        """
        score_diff = scores[:, None] - scores[..., None]
        normalized_score_diff = torch.sigmoid(score_diff / temperature)
        # set diagonal to 0
        normalized_score_diff = normalized_score_diff * (1 - torch.eye(scores.shape[1], device=scores.device))
        approx_ranks = normalized_score_diff.sum(-1) + 1
        return approx_ranks


class ApproxNDCG(ApproxLossFunction):
    """Approximate NDCG loss function for ranking tasks.

    Standard NDCG relies on non-differentiable sorting operations that prevent the use of gradient descent for direct
    optimization. Approximate NDCG overcomes this limitation by replacing the sorting step with a smooth,
    differentiable surrogate function that estimates the rank of each document based on its score. This approach allows
    the model to optimize a loss that is mathematically aligned with the final evaluation metric, reducing the mismatch
    between training objectives and testing performance.

    Originally proposed in: `Cumulated Gain-Based Evaluation of IR Techniques \
    <https://dl.acm.org/doi/10.1145/582415.582418>`_"""

    def __init__(self, temperature: float = 1, scale_gains: bool = True):
        """Initialize the ApproxNDCG loss function.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
            scale_gains (bool): Whether to scale the gains. Defaults to True.
        """
        super().__init__(temperature)
        self.scale_gains = scale_gains

    @staticmethod
    def get_dcg(
        ranks: torch.Tensor,
        targets: torch.Tensor,
        k: int | None = None,
        scale_gains: bool = True,
    ) -> torch.Tensor:
        """Compute the Discounted Cumulative Gain (DCG) for the given ranks and targets.

        Args:
            ranks (torch.Tensor): The ranks of the items.
            targets (torch.Tensor): The relevance scores of the items.
            k (int | None): Optional cutoff for the ranks. If provided, only computes DCG for the top k items.
            scale_gains (bool): Whether to scale the gains. Defaults to True.
        Returns:
            torch.Tensor: The computed DCG values.
        """
        log_ranks = torch.log2(1 + ranks)
        discounts = 1 / log_ranks
        if scale_gains:
            gains = 2**targets - 1
        else:
            gains = targets
        dcgs = gains * discounts
        if k is not None:
            dcgs = dcgs.masked_fill(ranks > k, 0)
        return dcgs.sum(dim=-1)

    @staticmethod
    def get_ndcg(
        ranks: torch.Tensor,
        targets: torch.Tensor,
        k: int | None = None,
        scale_gains: bool = True,
        optimal_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Normalized Discounted Cumulative Gain (NDCG) for the given ranks and targets.

        Args:
            ranks (torch.Tensor): The ranks of the items.
            targets (torch.Tensor): The relevance scores of the items.
            k (int | None): Cutoff for the ranks. If provided, only computes NDCG for the top k items. Defaults to None.
            scale_gains (bool): Whether to scale the gains. Defaults to True.
            optimal_targets (torch.Tensor | None): Optional tensor of optimal targets for normalization. If None, uses
                the targets. Defaults to None.
        Returns:
            torch.Tensor: The computed NDCG values.
        """
        targets = targets.clamp(min=0)
        if optimal_targets is None:
            optimal_targets = targets
        optimal_ranks = torch.argsort(torch.argsort(optimal_targets, descending=True))
        optimal_ranks = optimal_ranks + 1
        dcg = ApproxNDCG.get_dcg(ranks, targets, k, scale_gains)
        idcg = ApproxNDCG.get_dcg(optimal_ranks, optimal_targets, k, scale_gains)
        ndcg = dcg / (idcg.clamp(min=1e-12))
        return ndcg

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the ApproxNDCG loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        approx_ranks = self.get_approx_ranks(scores, self.temperature)
        ndcg = self.get_ndcg(approx_ranks, targets, k=None, scale_gains=self.scale_gains)
        loss = 1 - ndcg
        return loss.mean()


class ApproxMRR(ApproxLossFunction):
    """Approximate Mean Reciprocal Rank (MRR) loss function for ranking tasks.

    Mean Reciprocal Rank (MRR) is a metric used to evaluate ranking systems by focusing on the position of the
    first relevant result, making it ideal for tasks like question answering where the user wants one correct answer
    immediately. It assigns a score of $1/k$, where $k$ is the rank of the first relevant document; for example, if the
    correct result is at position 1, the score is 1, but if it is at position 10, the score drops to 0.1.  The final
    MRR is simply the average of these reciprocal scores across all queries in the dataset.
    Approximate MRR replaces the non-differentiable discrete ranking operation with a smooth, differentiable surrogate
    function based on pairwise score comparisons, enabling the model to directly maximize the reciprocal rank of the
    relevant document via gradient descent.
    """

    def __init__(self, temperature: float = 1):
        """Initialize the ApproxMRR loss function.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
        """
        super().__init__(temperature)

    @staticmethod
    def get_mrr(ranks: torch.Tensor, targets: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Compute the Mean Reciprocal Rank (MRR) for the given ranks and targets.

        Args:
            ranks (torch.Tensor): The ranks of the items.
            targets (torch.Tensor): The relevance scores of the items.
            k (int | None): Optional cutoff for the ranks. If provided, only computes MRR for the top k items.
        Returns:
            torch.Tensor: The computed MRR values.
        """
        targets = targets.clamp(None, 1)
        reciprocal_ranks = 1 / ranks
        mrr = reciprocal_ranks * targets
        if k is not None:
            mrr = mrr.masked_fill(ranks > k, 0)
        mrr = mrr.max(dim=-1)[0]
        return mrr

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the ApproxMRR loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        approx_ranks = self.get_approx_ranks(scores, self.temperature)
        mrr = self.get_mrr(approx_ranks, targets, k=None)
        loss = 1 - mrr
        return loss.mean()


class ApproxRankMSE(ApproxLossFunction):
    """Approximate Rank Mean Squared Error (RankMSE) loss function for ranking tasks.

    Rank Mean Squared Error (RankMSE) penalizes the squared differences between predicted document ranks and their
    ground truth ranks. Because standard discrete sorting prevents gradient descent, Approximate RankMSE uses a
    smooth, differentiable approximation of these ranks. It computes the Mean Squared Error between the continuous
    approximate ranks and the true target ranks, optionally applying position-based discounting (like log2 or
    reciprocal weights) to penalize errors at the top of the list more heavily.

    Originally proposed in: `Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs
    for Passage Re-ranking <https://link.springer.com/chapter/10.1007/978-3-031-88714-7_31>`_
    """

    def __init__(
        self,
        temperature: float = 1,
        discount: Literal["log2", "reciprocal"] | None = None,
    ):
        """Initialize the ApproxRankMSE loss function.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
            discount (Literal["log2", "reciprocal"] | None): Discounting strategy for the loss. If None, no discounting
                is applied. Defaults to None.
        """
        super().__init__(temperature)
        self.discount = discount

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the ApproxRankMSE loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        approx_ranks = self.get_approx_ranks(scores, self.temperature)
        ranks = torch.argsort(torch.argsort(targets, descending=True)) + 1
        loss = torch.nn.functional.mse_loss(approx_ranks, ranks.to(approx_ranks), reduction="none")
        if self.discount == "log2":
            weight = 1 / torch.log2(ranks + 1)
        elif self.discount == "reciprocal":
            weight = 1 / ranks
        else:
            weight = 1
        loss = loss * weight
        loss = loss.mean()
        return loss
