"""
Lightning IR module for loss functions.

This module provides various loss functions used in the Lightning IR framework, including in-batch losses, pairwise
losses, and regularization techniques.
"""

from .approximate import ApproxMRR, ApproxNDCG, ApproxRankMSE
from .in_batch import InBatchCrossEntropy, ScoreBasedInBatchCrossEntropy, ScoreBasedInBatchLossFunction
from .listwise import InfoNCE, KLDivergence, PearsonCorrelation
from .pairwise import ConstantMarginMSE, RankNet, SupervisedMarginMSE
from .regularization import FLOPSRegularization, L1Regularization, L2Regularization

__all__ = [
    "ApproxMRR",
    "ApproxNDCG",
    "ApproxRankMSE",
    "ConstantMarginMSE",
    "FLOPSRegularization",
    "InBatchCrossEntropy",
    "InfoNCE",
    "KLDivergence",
    "L1Regularization",
    "L2Regularization",
    "PearsonCorrelation",
    "RankNet",
    "ScoreBasedInBatchCrossEntropy",
    "ScoreBasedInBatchLossFunction",
    "SupervisedMarginMSE",
]
