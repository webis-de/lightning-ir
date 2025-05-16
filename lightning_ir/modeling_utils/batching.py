from functools import wraps
from typing import Callable

import torch


def _batch_scoring(
    similarity_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Helper function to batch similarity functions to avoid memory issues with large batch sizes or high numbers
    of documents per query."""
    BATCH_SIZE = 16384

    @wraps(similarity_function)
    def batch_similarity_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= BATCH_SIZE:
            return similarity_function(x, y)
        out = torch.zeros(x.shape[0], x.shape[1], y.shape[2], device=x.device, dtype=x.dtype)
        for i in range(0, x.shape[0], BATCH_SIZE):
            out[i : i + BATCH_SIZE] = similarity_function(x[i : i + BATCH_SIZE], y[i : i + BATCH_SIZE])
        return out

    return batch_similarity_function
