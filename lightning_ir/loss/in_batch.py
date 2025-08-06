"""
In-batch loss functions for the Lightning IR framework.

This module contains loss functions that operate on batches of data,
comparing examples within the same batch for training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

import torch

from .base import LossFunction

if TYPE_CHECKING:
    from ..base import LightningIROutput
    from ..data import TrainBatch


class InBatchLossFunction(LossFunction):
    """Base class for in-batch loss functions that compute in-batch indices for positive and negative samples."""

    def __init__(
        self,
        pos_sampling_technique: Literal["all", "first"] = "all",
        neg_sampling_technique: Literal["all", "first", "all_and_non_first"] = "all",
        max_num_neg_samples: int | None = None,
    ):
        """Initialize the InBatchLossFunction.

        Args:
            pos_sampling_technique (Literal["all", "first"]): Technique for positive sample sampling.
            neg_sampling_technique (Literal["all", "first", "all_and_non_first"]): Technique for negative sample
                sampling.
            max_num_neg_samples (int | None): Maximum number of negative samples to consider. If None, all negative
                samples are considered.
        Raises:
            ValueError: If the negative sampling technique is invalid for the given positive sampling technique.
        """
        super().__init__()
        self.pos_sampling_technique = pos_sampling_technique
        self.neg_sampling_technique = neg_sampling_technique
        self.max_num_neg_samples = max_num_neg_samples
        if self.neg_sampling_technique == "all_and_non_first" and self.pos_sampling_technique != "first":
            raise ValueError("all_and_non_first is only valid with pos_sampling_technique first")

    def _get_pos_mask(
        self,
        num_queries: int,
        num_docs: int,
        max_idx: torch.Tensor,
        min_idx: torch.Tensor,
        output: LightningIROutput,
        batch: TrainBatch,
    ) -> torch.Tensor:
        """Get the mask for positive samples based on the sampling technique.

        Args:
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            max_idx (torch.Tensor): Maximum index for each query.
            min_idx (torch.Tensor): Minimum index for each query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: A mask tensor indicating the positions of positive samples.
        Raises:
            ValueError: If the positive sampling technique is invalid.
        """
        if self.pos_sampling_technique == "all":
            pos_mask = torch.arange(num_queries * num_docs)[None].greater_equal(min_idx) & torch.arange(
                num_queries * num_docs
            )[None].less(max_idx)
        elif self.pos_sampling_technique == "first":
            pos_mask = torch.arange(num_queries * num_docs)[None].eq(min_idx)
        else:
            raise ValueError("invalid pos sampling technique")
        return pos_mask

    def _get_neg_mask(
        self,
        num_queries: int,
        num_docs: int,
        max_idx: torch.Tensor,
        min_idx: torch.Tensor,
        output: LightningIROutput,
        batch: TrainBatch,
    ) -> torch.Tensor:
        """Get the mask for negative samples based on the sampling technique.

        Args:
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            max_idx (torch.Tensor): Maximum index for each query.
            min_idx (torch.Tensor): Minimum index for each query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: A mask tensor indicating the positions of negative samples.
        Raises:
            ValueError: If the negative sampling technique is invalid.
        """
        if self.neg_sampling_technique == "all_and_non_first":
            neg_mask = torch.arange(num_queries * num_docs)[None].not_equal(min_idx)
        elif self.neg_sampling_technique == "all":
            neg_mask = torch.arange(num_queries * num_docs)[None].less(min_idx) | torch.arange(num_queries * num_docs)[
                None
            ].greater_equal(max_idx)
        elif self.neg_sampling_technique == "first":
            neg_mask = torch.arange(num_queries * num_docs)[None, None].eq(min_idx).any(1) & torch.arange(
                num_queries * num_docs
            )[None].ne(min_idx)
        else:
            raise ValueError("invalid neg sampling technique")
        return neg_mask

    def get_ib_idcs(self, output: LightningIROutput, batch: TrainBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get in-batch indices for positive and negative samples.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Indices of positive and negative samples.
        Raises:
            ValueError: If scores are not present in the output.
        """
        if output.scores is None:
            raise ValueError("Expected scores in LightningIROutput")
        num_queries, num_docs = output.scores.shape
        min_idx = torch.arange(num_queries)[:, None] * num_docs
        max_idx = min_idx + num_docs
        pos_mask = self._get_pos_mask(num_queries, num_docs, max_idx, min_idx, output, batch)
        neg_mask = self._get_neg_mask(num_queries, num_docs, max_idx, min_idx, output, batch)
        pos_idcs = pos_mask.nonzero(as_tuple=True)[1]
        neg_idcs = neg_mask.nonzero(as_tuple=True)[1]
        if self.max_num_neg_samples is not None:
            neg_idcs = neg_idcs.view(num_queries, -1)
            if neg_idcs.shape[-1] > 1:
                neg_idcs = neg_idcs[:, torch.randperm(neg_idcs.shape[-1])]
            neg_idcs = neg_idcs[:, : self.max_num_neg_samples]
            neg_idcs = neg_idcs.reshape(-1)
        return pos_idcs, neg_idcs


class ScoreBasedInBatchLossFunction(InBatchLossFunction):
    """Base class for in-batch loss functions that compute in-batch indices based on scores."""

    def __init__(self, min_target_diff: float, max_num_neg_samples: int | None = None):
        """Initialize the ScoreBasedInBatchLossFunction.

        Args:
            min_target_diff (float): Minimum target difference for negative sampling.
            max_num_neg_samples (int | None): Maximum number of negative samples.
        """
        super().__init__(
            pos_sampling_technique="first",
            neg_sampling_technique="all_and_non_first",
            max_num_neg_samples=max_num_neg_samples,
        )
        self.min_target_diff = min_target_diff

    def _sort_mask(
        self, mask: torch.Tensor, num_queries: int, num_docs: int, output: LightningIROutput, batch: TrainBatch
    ) -> torch.Tensor:
        """Sort the mask based on the scores and targets.

        Args:
            mask (torch.Tensor): The initial mask tensor.
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The sorted mask tensor.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        idcs = targets.argsort(descending=True).argsort().cpu()
        idcs = idcs + torch.arange(num_queries)[:, None] * num_docs
        block_idcs = torch.arange(num_docs)[None] + torch.arange(num_queries)[:, None] * num_docs
        return mask.scatter(1, block_idcs, mask.gather(1, idcs))

    def _get_pos_mask(
        self,
        num_queries: int,
        num_docs: int,
        max_idx: torch.Tensor,
        min_idx: torch.Tensor,
        output: LightningIROutput,
        batch: TrainBatch,
    ) -> torch.Tensor:
        """Get the mask for positive samples and sort it based on scores.

        Args:
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            max_idx (torch.Tensor): Maximum index for each query.
            min_idx (torch.Tensor): Minimum index for each query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: A mask tensor indicating the positions of positive samples, sorted by scores.
        """
        pos_mask = super()._get_pos_mask(num_queries, num_docs, max_idx, min_idx, output, batch)
        pos_mask = self._sort_mask(pos_mask, num_queries, num_docs, output, batch)
        return pos_mask

    def _get_neg_mask(
        self,
        num_queries: int,
        num_docs: int,
        max_idx: torch.Tensor,
        min_idx: torch.Tensor,
        output: LightningIROutput,
        batch: TrainBatch,
    ) -> torch.Tensor:
        """Get the mask for negative samples and sort it based on scores.

        Args:
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            max_idx (torch.Tensor): Maximum index for each query.
            min_idx (torch.Tensor): Minimum index for each query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: A mask tensor indicating the positions of negative samples, sorted by scores.
        """
        neg_mask = super()._get_neg_mask(num_queries, num_docs, max_idx, min_idx, output, batch)
        neg_mask = self._sort_mask(neg_mask, num_queries, num_docs, output, batch)
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch).cpu()
        max_score, _ = targets.max(dim=-1, keepdim=True)
        score_diff = (max_score - targets).cpu()
        score_mask = score_diff.ge(self.min_target_diff)
        block_idcs = torch.arange(num_docs)[None] + torch.arange(num_queries)[:, None] * num_docs
        neg_mask = neg_mask.scatter(1, block_idcs, score_mask)
        # num_neg_samples might be different between queries
        num_neg_samples = neg_mask.sum(dim=1)
        min_num_neg_samples = num_neg_samples.min()
        additional_neg_samples = num_neg_samples - min_num_neg_samples
        for query_idx, neg_samples in enumerate(additional_neg_samples):
            neg_idcs = neg_mask[query_idx].nonzero().squeeze(1)
            additional_neg_idcs = neg_idcs[torch.randperm(neg_idcs.shape[0])][:neg_samples]
            assert neg_mask[query_idx, additional_neg_idcs].all()
            neg_mask[query_idx, additional_neg_idcs] = False
            assert neg_mask[query_idx].sum().eq(min_num_neg_samples)
        return neg_mask


class InBatchCrossEntropy(InBatchLossFunction):
    """In-batch cross-entropy loss function for ranking tasks.
    Originally proposed in: `Fast Single-Class Classification and the Principle of Logit Separation
    <https://arxiv.org/pdf/1705.10246v1>`_"""

    def compute_loss(self, output: LightningIROutput) -> torch.Tensor:
        """Compute the in-batch cross-entropy loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss


class ScoreBasedInBatchCrossEntropy(ScoreBasedInBatchLossFunction):
    """In-batch cross-entropy loss function based on scores for ranking tasks."""

    def compute_loss(self, output: LightningIROutput) -> torch.Tensor:
        """Compute the in-batch cross-entropy loss based on scores.

        Args:
            output (LightningIROutput): The output from the model containing scores.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss
