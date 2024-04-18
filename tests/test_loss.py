from typing import Type

import pytest
import torch

from lightning_ir.loss.loss import (
    ConstantMarginMSE,
    InBatchCrossEntropy,
    InBatchLossFunction,
    KLDivergence,
    LossFunction,
    RankNet,
    SupervisedMarginMSE,
)

torch.manual_seed(42)


@pytest.fixture(scope="module")
def batch_size() -> int:
    return 4


@pytest.fixture(scope="module")
def depth() -> int:
    return 10


@pytest.fixture(scope="module")
def logits(batch_size: int, depth: int) -> torch.Tensor:
    tensor = torch.randn((batch_size, depth), requires_grad=True)
    return tensor


@pytest.fixture(scope="module")
def labels(batch_size: int, depth: int) -> torch.Tensor:
    tensor = torch.randint(0, 5, (batch_size, depth))
    return tensor


@pytest.mark.parametrize(
    "LossFunc",
    [
        ConstantMarginMSE,
        KLDivergence,
        RankNet,
        SupervisedMarginMSE,
    ],
)
def test_loss_func(
    logits: torch.Tensor,
    labels: torch.Tensor,
    LossFunc: Type[LossFunction],
):
    loss_func = LossFunc()
    loss = loss_func.compute_loss(logits, labels)
    assert loss >= 0
    assert loss.requires_grad


@pytest.mark.parametrize(
    "InBatchLossFunc",
    [
        InBatchCrossEntropy,
    ],
)
def test_in_batch_loss_func(
    InBatchLossFunc: Type[InBatchLossFunction], logits: torch.Tensor
):
    loss_func = InBatchLossFunc()
    loss = loss_func.compute_loss(logits)
    assert loss >= 0
    assert loss.requires_grad
