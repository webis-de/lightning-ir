import pytest

from mvr.tide import TideModel, TideModule, TideConfig
from mvr.datamodule import MVRDataModule
from mvr.loss import LocalizedContrastive, MarginMSE, RankNet


@pytest.fixture(scope="module")
def tide_model(model_name_or_path: str) -> TideModel:
    config = TideConfig.from_pretrained(model_name_or_path)
    model = TideModel.from_pretrained(model_name_or_path, config=config)
    return model


@pytest.fixture(scope="module")
def margin_mse_module(model_name_or_path: str) -> TideModule:
    config = TideConfig.from_pretrained(model_name_or_path)
    return TideModule(model_name_or_path, config, MarginMSE())


@pytest.fixture(scope="module")
def ranknet_module(model_name_or_path: str) -> TideModule:
    config = TideConfig.from_pretrained(model_name_or_path)
    return TideModule(model_name_or_path, config, RankNet())


@pytest.fixture(scope="module")
def localized_contrastive_module(model_name_or_path: str) -> TideModule:
    config = TideConfig.from_pretrained(model_name_or_path)
    return TideModule(model_name_or_path, config, LocalizedContrastive())


def test_tide_margin_mse(
    margin_mse_module: TideModule, tuples_datamodule: MVRDataModule
):
    dataloader = tuples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = margin_mse_module.training_step(batch, 0)
    assert loss


def test_tide_ranknet(ranknet_module: TideModule, rank_run_datamodule: MVRDataModule):
    dataloader = rank_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = ranknet_module.training_step(batch, 0)
    assert loss


def test_tide_localized_contrastive(
    localized_contrastive_module: TideModule,
    single_relevant_run_datamodule: MVRDataModule,
):
    dataloader = single_relevant_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = localized_contrastive_module.training_step(batch, 0)
    assert loss


def test_seralize_deserialize(
    tide_model: TideModel, tmpdir_factory: pytest.TempdirFactory
):
    save_dir = tmpdir_factory.mktemp("tide")
    tide_model.save_pretrained(save_dir)
    new_model = TideModel.from_pretrained(save_dir, mask_punctuation=False)
    for key, value in tide_model.config.__dict__.items():
        if key in (
            "torch_dtype",
            "_name_or_path",
            "_commit_hash",
            "transformers_version",
            "model_type",
        ):
            continue
        if key == "mask_punctuation":
            assert value and not getattr(new_model.config, key)
            continue
        assert getattr(new_model.config, key) == value
    for key, value in tide_model.state_dict().items():
        assert new_model.state_dict()[key].equal(value)
