"""
Base classes and abstract interfaces for loss functions in the Lightning IR framework.

This module defines the abstract base classes and common functionality for all loss functions
used in the Lightning IR framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

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
