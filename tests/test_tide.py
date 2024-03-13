import pytest

from mvr.tide import TideModel, TideModule, TideConfig
from mvr.datamodule import MVRDataModule
from mvr.loss import SupervisedMarginMSE
from mvr.datamodule import TupleDatasetConfig


@pytest.fixture(scope="module")
def tide_model(model_name_or_path: str) -> TideModel:
    config = TideConfig.from_pretrained(model_name_or_path)
    model = TideModel.from_pretrained(model_name_or_path, config=config)
    return model


@pytest.fixture(scope="module")
def datamodule(model_name_or_path: str, tide_model: TideModel) -> MVRDataModule:
    datamodule = MVRDataModule(
        model_name_or_path=model_name_or_path,
        config=tide_model.config,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset="msmarco-passage/train/colbert-docpairs",
        train_dataset_config=TupleDatasetConfig(4),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="module")
def tide_module(model_name_or_path: str) -> TideModule:
    config = TideConfig.from_pretrained(model_name_or_path)
    return TideModule(model_name_or_path, config, SupervisedMarginMSE())


def test_training_step(tide_module: TideModule, datamodule: MVRDataModule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = tide_module.training_step(batch, 0)
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
