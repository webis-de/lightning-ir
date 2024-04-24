from functools import lru_cache
from pathlib import Path
from typing import Sequence, Union

import pytest
from _pytest.fixtures import SubRequest
from transformers import BertModel

from lightning_ir.cross_encoder.model import CrossEncoderConfig, CrossEncoderModel
from lightning_ir.cross_encoder.module import CrossEncoderModule
from lightning_ir.cross_encoder.mono import (
    MonoBertModel,
    MonoBertModule,
    MonoElectraModel,
    MonoElectraModule,
    MonoRobertaModel,
    MonoRobertaModule,
)
from lightning_ir.data.datamodule import (
    LightningIRDataModule,
    RunDatasetConfig,
    TupleDatasetConfig,
)
from lightning_ir.loss.loss import (
    ConstantMarginMSE,
    LossFunction,
    RankNet,
    SupervisedMarginMSE,
)

DATA_DIR = Path(__file__).parent / "data"


class TestModel(CrossEncoderModel):
    config_class = CrossEncoderConfig

    def __init__(self, config: CrossEncoderConfig) -> None:
        config.num_hidden_layers = 1
        super().__init__(config, "bert")
        self.bert = BertModel.from_pretrained(
            config.name_or_path, config=config, add_pooling_layer=False
        )


class TestModule(CrossEncoderModule):
    def __init__(
        self,
        model: TestModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, CrossEncoderConfig):
                    raise ValueError(
                        "To initialize a new model pass a CrossEncoderConfig."
                    )
                model = TestModel(config)
            else:
                model = TestModel.from_pretrained(model_name_or_path, config=config)
        super().__init__(model, loss_functions, evaluation_metrics)


MODULE_MAP = {
    TestModel: TestModule,
    MonoBertModel: MonoBertModule,
    MonoElectraModel: MonoElectraModule,
    MonoRobertaModel: MonoRobertaModule,
}
MODELS = Union[TestModel, MonoBertModel, MonoElectraModel, MonoRobertaModel]
MODULES = Union[TestModule, MonoBertModule, MonoElectraModule, MonoRobertaModule]
MODEL_NAME_OR_PATH_MAP = {
    TestModule: "sentence-transformers/all-MiniLM-L6-v2",
    MonoBertModule: "sentence-transformers/all-MiniLM-L6-v2",
    MonoElectraModule: "google/electra-small-discriminator",
    MonoRobertaModule: "FacebookAI/roberta-base",
    TestModel: "sentence-transformers/all-MiniLM-L6-v2",
    MonoBertModel: "sentence-transformers/all-MiniLM-L6-v2",
    MonoElectraModel: "google/electra-small-discriminator",
    MonoRobertaModel: "FacebookAI/roberta-base",
}


@pytest.fixture(scope="module", params=list(MODULE_MAP.keys()))
def model(request: SubRequest) -> MODELS:
    Model = request.param
    model_name_or_path = MODEL_NAME_OR_PATH_MAP[request.param]
    config = Model.config_class.from_pretrained(model_name_or_path, num_hidden_layers=1)
    _model = Model.from_pretrained(model_name_or_path, config=config)
    return _model


@pytest.fixture(
    scope="module",
    params=[ConstantMarginMSE(), RankNet(), SupervisedMarginMSE()],
)
def module(model: MODELS, request: SubRequest) -> MODULES:
    loss_function = request.param
    module = MODULE_MAP[type(model)](
        model, loss_functions=[loss_function], evaluation_metrics=["nDCG@10", "loss"]
    )
    return module


@lru_cache
def tuples_datamodule(model: MODELS) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        model_name_or_path=model.config.name_or_path,
        config=model.config,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset="msmarco-passage/train/kd-docpairs",
        train_dataset_config=TupleDatasetConfig("score", 2),
        inference_datasets=[
            str(DATA_DIR / "clueweb09-en-trec-web-2009-diversity.jsonl"),
            str(DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run"),
        ],
        inference_dataset_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


def test_training_step(module: MODULES):
    datamodule = tuples_datamodule(module.model)
    if not isinstance(datamodule.config, CrossEncoderConfig):
        pytest.skip()
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = module.training_step(batch, 0)
    assert loss


def test_validation(module: MODULES):
    datamodule = tuples_datamodule(module.model)
    if not isinstance(datamodule.config, CrossEncoderConfig):
        pytest.skip()
    dataloader = datamodule.val_dataloader()[0]
    for batch, batch_idx in zip(dataloader, range(2)):
        module.validation_step(batch, batch_idx, 0)

    metrics = module.on_validation_epoch_end()
    assert metrics is not None
    for key, value in metrics.items():
        metric = key.split("/")[1]
        assert metric in {"nDCG@10"} or "validation" in metric
        assert value


def test_seralize_deserialize(model: MODELS, tmpdir_factory: pytest.TempdirFactory):
    save_dir = tmpdir_factory.mktemp(model.config_class.model_type)
    model.save_pretrained(save_dir)
    kwargs = {}
    new_model = type(model).from_pretrained(save_dir, **kwargs)
    for key, value in model.config.__dict__.items():
        if key in (
            "torch_dtype",
            "_name_or_path",
            "_commit_hash",
            "transformers_version",
            "model_type",
        ):
            continue
        assert getattr(new_model.config, key) == value
    for key, value in model.state_dict().items():
        assert new_model.state_dict()[key].equal(value)
