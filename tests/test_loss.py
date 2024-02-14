import pytest
import torch
from typing import Literal, Tuple, Type

from tide.loss import (
    LocalizedContrastive,
    MarginMSE,
    RankNet,
    KLDivergence,
    PAD_VALUE,
    LossFunction,
)

torch.manual_seed(42)


@pytest.fixture(scope="module")
def shape():
    return (32, 64)


@pytest.fixture(scope="module")
def logits(shape: Tuple[int, int]):
    tensor = torch.randn(shape)
    tensor[torch.rand(shape) < 0.1] = PAD_VALUE
    return tensor


@pytest.fixture(scope="module")
def labels(shape: Tuple[int, int]):
    tensor = torch.randint(0, 5, shape)
    tensor[torch.rand(shape) < 0.1] = PAD_VALUE
    return tensor


@pytest.mark.parametrize("in_batch_negatives", [True, False])
@pytest.mark.parametrize("reduction", ["mean", "sum", None])
@pytest.mark.parametrize(
    "LossFunc", [LocalizedContrastive, MarginMSE, RankNet, KLDivergence]
)
def test_loss_func(
    LossFunc: Type[LossFunction],
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: Literal["mean", "sum"] | None,
    in_batch_negatives: bool,
):
    loss_func = LossFunc(reduction=reduction, in_batch_negatives=in_batch_negatives)
    loss = loss_func.compute_loss(logits, labels)
    if loss_func.in_batch_negatives:
        if reduction is None:
            return
        logits = logits.repeat(1, logits.shape[0])
        ib_loss = loss_func.compute_in_batch_negative_loss(logits)
        loss = loss + ib_loss
    if reduction is None:
        assert loss.shape[0] == logits.shape[0]
        assert loss.mean()
    else:
        assert loss
