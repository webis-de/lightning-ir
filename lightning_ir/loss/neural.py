"""
Neural sorting-based loss functions for the Lightning IR framework.

This module contains loss functions that use neural sorting techniques
to compute differentiable ranking-based losses.
"""

from __future__ import annotations

import torch

from .base import ListwiseLossFunction


class NeuralLossFunction(ListwiseLossFunction):
    """Base class for neural loss functions that compute ranks from scores using neural sorting."""

    # TODO add neural loss functions

    def __init__(self, temperature: float = 1, tol: float = 1e-5, max_iter: int = 50) -> None:
        """Initialize the NeuralLossFunction.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
            tol (float): Tolerance for convergence. Defaults to 1e-5.
            max_iter (int): Maximum number of iterations for convergence. Defaults to 50.
        """
        super().__init__()
        self.temperature = temperature
        self.tol = tol
        self.max_iter = max_iter

    def neural_sort(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute the neural sort permutation matrix from scores.

        Args:
            scores (torch.Tensor): The input scores tensor.
        Returns:
            torch.Tensor: The computed permutation matrix.
        """
        # https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py
        scores = scores.unsqueeze(-1)
        dim = scores.shape[1]
        one = torch.ones((dim, 1), device=scores.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = dim + 1 - 2 * (torch.arange(dim, device=scores.device) + 1)
        C = torch.matmul(scores, scaling.to(scores).unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        P_hat = torch.nn.functional.softmax(P_max / self.temperature, dim=-1)

        P_hat = self.sinkhorn_scaling(P_hat)

        return P_hat

    def sinkhorn_scaling(self, mat: torch.Tensor) -> torch.Tensor:
        """Apply Sinkhorn scaling to the permutation matrix.

        Args:
            mat (torch.Tensor): The input permutation matrix.
        Returns:
            torch.Tensor: The scaled permutation matrix.
        """
        # https://github.com/allegro/allRank/blob/master/allrank/models/losses/loss_utils.py#L8
        idx = 0
        while True:
            if (
                torch.max(torch.abs(mat.sum(dim=2) - 1.0)) < self.tol
                and torch.max(torch.abs(mat.sum(dim=1) - 1.0)) < self.tol
            ) or idx > self.max_iter:
                break
            mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1e-12)
            mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=1e-12)
            idx += 1

        return mat

    def get_sorted_targets(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get the sorted targets based on the neural sort permutation matrix.

        Args:
            scores (torch.Tensor): The input scores tensor.
            targets (torch.Tensor): The targets tensor.
        Returns:
            torch.Tensor: The sorted targets tensor.
        """
        permutation_matrix = self.neural_sort(scores)
        pred_sorted_targets = torch.matmul(permutation_matrix, targets[..., None].to(permutation_matrix)).squeeze(-1)
        return pred_sorted_targets
