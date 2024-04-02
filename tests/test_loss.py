from typing import Type

import pytest
import torch

from mvr.loss import (
    ConstantMarginMSE,
    DocMarginMSE,
    KLDivergence,
    LossFunction,
    RankNet,
    SupervisedMarginMSE,
    QueryLevelCrossEntropy,
)
from mvr.mvr import MVRConfig, ScoringFunction

torch.manual_seed(42)


@pytest.fixture(scope="module")
def num_queries() -> int:
    return 4


@pytest.fixture(scope="module")
def num_docs() -> int:
    return 10


@pytest.fixture(scope="module")
def query_embeddings(num_queries: int):
    tensor = torch.randn(num_queries, 32, 128, requires_grad=True)
    return tensor


@pytest.fixture(scope="module")
def doc_embeddings(num_queries: int, num_docs: int):
    tensor = torch.randn(num_queries * num_docs, 64, 128, requires_grad=True)
    return tensor


@pytest.fixture(scope="module")
def labels(num_queries: int, num_docs: int):
    tensor = torch.randint(0, 5, (num_queries * num_docs,))
    return tensor


@pytest.mark.parametrize(
    "LossFunc",
    [
        ConstantMarginMSE,
        DocMarginMSE,
        KLDivergence,
        RankNet,
        SupervisedMarginMSE,
        QueryLevelCrossEntropy,
    ],
)
def test_loss_func(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    labels: torch.Tensor,
    LossFunc: Type[LossFunction],
):
    query_scoring_mask = torch.ones(query_embeddings.shape[:-1])
    doc_scoring_mask = torch.ones(doc_embeddings.shape[:-1])
    loss_func = LossFunc()
    loss_func.set_scoring_function(ScoringFunction(MVRConfig()))
    loss = loss_func.compute_loss(
        query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask, labels
    )
    for _, value in loss.items():
        assert value >= 0
        assert value.requires_grad
