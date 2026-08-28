"""
Module containing Lightning IR callbacks.

This module provides PyTorch Lightning compatible callbacks for various stages and tasks.

Callbacks let you to customize the training and evaluation process in PyTorch Lightning, allowing you to
execute code at specific points during the training loop, such as after each epoch or batch.
"""

from .callbacks import (
    IndexCallback,
    MeanValidationMetricCallback,
    MvrViewCollapseCallback,
    RankCallback,
    RegisterLocalDatasetCallback,
    ReRankCallback,
    SearchCallback,
    SparsityCollapseCallback,
)

__all__ = [
    "IndexCallback",
    "MeanValidationMetricCallback",
    "MvrViewCollapseCallback",
    "RankCallback",
    "RegisterLocalDatasetCallback",
    "ReRankCallback",
    "SearchCallback",
    "SparsityCollapseCallback",
]
