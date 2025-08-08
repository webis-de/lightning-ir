"""
Base classes and abstract interfaces for loss functions in the Lightning IR framework.

This module defines the abstract base classes and common functionality for all loss functions
used in the Lightning IR framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Tuple

import torch

if TYPE_CHECKING:
    from ..base import LightningIROutput
    from ..bi_encoder import BiEncoderOutput
    from ..data import TrainBatch


class LossFunction(ABC):
    """Base class for loss functions in the Lightning IR framework."""

    @abstractmethod
    def compute_loss(self, output: LightningIROutput, *args, **kwargs) -> torch.Tensor:
        """Compute the loss for the given output.

        Args:
            output (LightningIROutput): The output from the model.
        Returns:
            torch.Tensor: The computed loss.
        """
        ...

    def process_scores(self, output: LightningIROutput) -> torch.Tensor:
        """Process the scores from the output.

        Args:
            output (LightningIROutput): The output from the model.
        Returns:
            torch.Tensor: The scores tensor.
        """
        if output.scores is None:
            raise ValueError("Expected scores in LightningIROutput")
        return output.scores

    def process_targets(self, scores: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        """Process the targets from the batch.

        Args:
            scores (torch.Tensor): The scores tensor.
            batch (TrainBatch): The training batch.
        Returns:
            torch.Tensor: The processed targets tensor.
        """
        targets = batch.targets
        if targets is None:
            raise ValueError("Expected targets in TrainBatch")
        if targets.ndim > scores.ndim:
            return targets.amax(-1)
        return targets


class ScoringLossFunction(LossFunction):
    """Base class for loss functions that operate on scores."""

    @abstractmethod
    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the loss based on the scores and targets in the output and batch.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        ...


class EmbeddingLossFunction(LossFunction):
    """Base class for loss functions that operate on embeddings."""

    @abstractmethod
    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the loss based on the embeddings in the output.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        ...


class RegularizationLossFunction(EmbeddingLossFunction):
    """Base class for regularization loss functions that operate on embeddings."""

    def __init__(self, query_weight: float = 1e-4, doc_weight: float = 1e-4) -> None:
        """Initialize the RegularizationLossFunction.

        Args:
            query_weight (float): Weight for the query embeddings regularization. Defaults to 1e-4.
            doc_weight (float): Weight for the document embeddings regularization. Defaults to 1e-4.
        """
        self.query_weight = query_weight
        self.doc_weight = doc_weight

    def process_embeddings(self, output: BiEncoderOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process the embeddings from the output.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The processed query and document embeddings.
        Raises:
            ValueError: If query_embeddings are not present in the output.
            ValueError: If doc_embeddings are not present in the output.
        """
        query_embeddings = output.query_embeddings
        doc_embeddings = output.doc_embeddings
        if query_embeddings is None:
            raise ValueError("Expected query_embeddings in BiEncoderOutput")
        if doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")
        return query_embeddings.embeddings, doc_embeddings.embeddings


class PairwiseLossFunction(ScoringLossFunction):
    """Base class for pairwise loss functions."""

    def get_pairwise_idcs(self, targets: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Get pairwise indices for positive and negative samples based on targets.

        Args:
            targets (torch.Tensor): The targets tensor containing relevance labels.
        Returns:
            Tuple[torch.Tensor, ...]: Indices of positive and negative samples.
        """
        # positive items are items where label is greater than other label in sample
        return torch.nonzero(targets[..., None] > targets[:, None], as_tuple=True)


class ListwiseLossFunction(ScoringLossFunction):
    """Base class for listwise loss functions."""

    pass


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
