from functools import wraps
from typing import Callable

import torch


def _batch_elementwise_scoring(
    similarity_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Helper function to batch similarity functions to avoid memory issues with large batch sizes or high numbers
    of documents per query."""
    BATCH_SIZE = 16384

    @wraps(similarity_function)
    def batch_similarity_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x and y have shape (N, ..., D) where ... are broadcastable dimensions and D is the embedding dimension."""
        if x.shape[0] <= BATCH_SIZE:
            return similarity_function(x, y)
        out = torch.zeros(x.shape[0], x.shape[1], y.shape[2], device=x.device, dtype=x.dtype)
        for i in range(0, x.shape[0], BATCH_SIZE):
            out[i : i + BATCH_SIZE] = similarity_function(x[i : i + BATCH_SIZE], y[i : i + BATCH_SIZE])
        return out

    return batch_similarity_function


def _batch_pairwise_scoring(
    similarity_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Helper function to batch similarity functions to avoid memory issues with large batch sizes or high numbers
    of documents per query."""
    BATCH_SIZE = 8192

    @wraps(similarity_function)
    def batch_similarity_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x has shape (N, D) and y has shape (M, D)"""
        if x.shape[0] <= BATCH_SIZE and y.shape[0] <= BATCH_SIZE:
            return similarity_function(x, y)
        out = torch.zeros(x.shape[0], y.shape[0], device=x.device, dtype=x.dtype)
        for i in range(0, x.shape[0], BATCH_SIZE):
            for j in range(0, y.shape[0], BATCH_SIZE):
                out[i : i + BATCH_SIZE, j : j + BATCH_SIZE] = similarity_function(
                    x[i : i + BATCH_SIZE], y[j : j + BATCH_SIZE]
                )
        return out

    return batch_similarity_function
