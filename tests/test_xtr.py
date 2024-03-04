import pytest

from mvr.xtr import XTRModel, XTRModule, XTRConfig
from mvr.datamodule import MVRDataModule
from mvr.loss import LocalizedContrastive, MarginMSE, RankNet


@pytest.fixture(scope="module")
def xtr_model(model_name_or_path: str) -> XTRModel:
    config = XTRConfig.from_pretrained(model_name_or_path, token_retrieval_k=10)
    model = XTRModel.from_pretrained(model_name_or_path, config=config)
    return model


@pytest.fixture(scope="module")
def margin_mse_module(model_name_or_path: str) -> XTRModule:
    config = XTRConfig.from_pretrained(model_name_or_path, token_retrieval_k=10)
    return XTRModule(model_name_or_path, config, MarginMSE())


@pytest.fixture(scope="module")
def ranknet_module(model_name_or_path: str) -> XTRModule:
    config = XTRConfig.from_pretrained(model_name_or_path, token_retrieval_k=10)
    return XTRModule(model_name_or_path, config, RankNet())


@pytest.fixture(scope="module")
def localized_contrastive_module(model_name_or_path: str) -> XTRModule:
    config = XTRConfig.from_pretrained(model_name_or_path, token_retrieval_k=10)
    return XTRModule(model_name_or_path, config, LocalizedContrastive())


def test_xtr_margin_mse(margin_mse_module: XTRModule, tuples_datamodule: MVRDataModule):
    dataloader = tuples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = margin_mse_module.training_step(batch, 0)
    assert loss


def test_xtr_ranknet(ranknet_module: XTRModule, rank_run_datamodule: MVRDataModule):
    dataloader = rank_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = ranknet_module.training_step(batch, 0)
    assert loss


def test_xtr_localized_contrastive(
    localized_contrastive_module: XTRModule,
    single_relevant_run_datamodule: MVRDataModule,
):
    dataloader = single_relevant_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = localized_contrastive_module.training_step(batch, 0)
    assert loss


def test_seralize_deserialize(
    xtr_model: XTRModel, tmpdir_factory: pytest.TempdirFactory
):
    save_dir = tmpdir_factory.mktemp("xtr")
    xtr_model.save_pretrained(save_dir)
    new_model = XTRModel.from_pretrained(save_dir, mask_punctuation=False)
    for key, value in xtr_model.config.__dict__.items():
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
    for key, value in xtr_model.state_dict().items():
        assert new_model.state_dict()[key].equal(value)
