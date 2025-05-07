import pytest
import torch

from lightning_ir.base.model import LightningIROutput
from lightning_ir.bi_encoder.bi_encoder_model import BiEncoderEmbedding, BiEncoderOutput
from lightning_ir.data.data import TrainBatch
from lightning_ir.loss.loss import (
    ApproxMRR,
    ApproxNDCG,
    ApproxRankMSE,
    ConstantMarginMSE,
    FLOPSRegularization,
    InBatchCrossEntropy,
    InBatchLossFunction,
    InfoNCE,
    KLDivergence,
    L1Regularization,
    L2Regularization,
    PearsonCorrelation,
    RankNet,
    RegularizationLossFunction,
    ScoreBasedInBatchCrossEntropy,
    ScoringLossFunction,
    SupervisedMarginMSE,
)

torch.manual_seed(42)


@pytest.fixture(scope="module")
def batch_size() -> int:
    return 4


@pytest.fixture(scope="module")
def sequence_length() -> int:
    return 8


@pytest.fixture(scope="module")
def depth() -> int:
    return 10


@pytest.fixture(scope="module")
def embedding_dim() -> int:
    return 4


@pytest.fixture(scope="module")
def output(batch_size: int, depth: int) -> LightningIROutput:
    return LightningIROutput(torch.randn((batch_size, depth), requires_grad=True))


@pytest.fixture(scope="module")
def targets(batch_size: int, depth: int) -> torch.Tensor:
    tensor = torch.randint(0, 5, (batch_size, depth))
    return tensor


@pytest.fixture(scope="module")
def embeddings(batch_size: int, sequence_length: int, embedding_dim: int) -> torch.Tensor:
    tensor = torch.randn((batch_size, sequence_length, embedding_dim), requires_grad=True)
    return tensor


@pytest.fixture(scope="module")
def batch(batch_size: int, depth: int, targets: torch.Tensor) -> TrainBatch:
    return TrainBatch(
        queries=["query"] * batch_size,
        docs=[[f"doc{i}" for i in range(depth)]] * batch_size,
        targets=targets,
    )


@pytest.mark.parametrize(
    "loss_func",
    [
        ApproxMRR(),
        ApproxNDCG(),
        ApproxRankMSE(),
        ConstantMarginMSE(),
        KLDivergence(),
        InfoNCE(),
        RankNet(),
        SupervisedMarginMSE(),
        PearsonCorrelation(),
    ],
    ids=[
        "ApproxMRR",
        "ApproxNDCG",
        "ApproxRankMSE",
        "ConstantMarginMSE",
        "KLDivergence",
        "InfoNCE",
        "RankNet",
        "SupervisedMarginMSE",
        "PearsonCorrelation",
    ],
)
def test_loss_func(output: LightningIROutput, batch: TrainBatch, loss_func: ScoringLossFunction):
    loss = loss_func.compute_loss(output, batch)
    assert loss >= 0
    assert loss.requires_grad


@pytest.mark.parametrize(
    "loss_func",
    [
        InBatchCrossEntropy(),
        InBatchCrossEntropy("first", "all_and_non_first"),
        ScoreBasedInBatchCrossEntropy(2),
    ],
    ids=["InBatchCrossEntropy(default)", "InBatchCrossEntropy(all_and_non_first)", "ScoreBasedInBatchCrossEntropy"],
)
def test_in_batch_loss_func(loss_func: InBatchLossFunction, output: LightningIROutput, batch: TrainBatch):
    pos_idcs, neg_idcs = loss_func.get_ib_idcs(output, batch)
    assert pos_idcs is not None
    assert neg_idcs is not None
    loss = loss_func.compute_loss(output)
    assert loss >= 0
    assert loss.requires_grad


@pytest.mark.parametrize(
    "loss_func", [L1Regularization(), L2Regularization(), FLOPSRegularization()], ids=["L1", "L2", "FLOPS"]
)
def test_regularization_loss_func(loss_func: RegularizationLossFunction, embeddings: torch.Tensor):
    loss = loss_func.compute_loss(
        BiEncoderOutput(
            None,
            BiEncoderEmbedding(embeddings, torch.empty(0), None),
            BiEncoderEmbedding(embeddings, torch.empty(0), None),
        )
    )
    assert loss >= 0
    assert loss.requires_grad
