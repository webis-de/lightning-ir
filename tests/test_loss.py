from typing import Literal, Tuple, Type

import pytest
import torch

from tide.loss import (
    PAD_VALUE,
    RankHinge,
    KLDivergence,
    LocalizedContrastive,
    LossFunction,
    MarginMSE,
    RankNet,
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


@pytest.mark.parametrize("in_batch_loss", ["ce", "hinge", None])
@pytest.mark.parametrize("reduction", ["mean", "sum", None])
@pytest.mark.parametrize(
    "LossFunc", [LocalizedContrastive, MarginMSE, RankNet, KLDivergence, RankHinge]
)
def test_loss_func(
    LossFunc: Type[LossFunction],
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: Literal["mean", "sum"] | None,
    in_batch_loss: Literal["ce", "hinge"] | None,
):
    loss_func = LossFunc(reduction=reduction, in_batch_loss=in_batch_loss)
    loss = loss_func.compute_loss(logits, labels)
    ib_logits = logits[:, [0]].repeat(1, logits.shape[0])
    ib_loss = loss_func.compute_in_batch_loss(ib_logits)
    if reduction is not None:
        loss = loss + ib_loss
    if reduction is None:
        assert loss.shape[0] == logits.shape[0]
        assert loss.mean()
    else:
        assert loss
    if in_batch_loss:
        if reduction is None:
            assert ib_loss.mean()
        else:
            assert ib_loss
    else:
        assert not ib_loss
