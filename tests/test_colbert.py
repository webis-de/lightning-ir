import pytest

from tide.colbert import ColBERTModel, ColBERTModule
from tide.datamodule import DataModule
from tide.loss import LocalizedContrastive, MarginMSE, RankNet
from tide.mvr import MVRConfig


@pytest.fixture(scope="module")
def colbert_model(model_name_or_path: str) -> ColBERTModel:
    return ColBERTModel(model_name_or_path, MVRConfig(query_expansion=True))


@pytest.fixture(scope="module")
def margin_mse_module(colbert_model: ColBERTModel) -> ColBERTModule:
    return ColBERTModule(colbert_model, MarginMSE())


@pytest.fixture(scope="module")
def ranknet_module(colbert_model: ColBERTModel) -> ColBERTModule:
    return ColBERTModule(colbert_model, RankNet())


@pytest.fixture(scope="module")
def localized_contrastive_module(colbert_model: ColBERTModel) -> ColBERTModule:
    return ColBERTModule(colbert_model, LocalizedContrastive())


def test_colbert_margin_mse(
    margin_mse_module: ColBERTModule, triples_datamodule: DataModule
):
    dataloader = triples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = margin_mse_module.training_step(batch, 0)
    assert loss


def test_colbert_ranknet(
    ranknet_module: ColBERTModule, rank_run_datamodule: DataModule
):
    dataloader = rank_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = ranknet_module.training_step(batch, 0)
    assert loss


def test_colbert_localized_contrastive(
    localized_contrastive_module: ColBERTModule,
    single_relevant_run_datamodule: DataModule,
):
    dataloader = single_relevant_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = localized_contrastive_module.training_step(batch, 0)
    assert loss
