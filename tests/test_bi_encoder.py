from pathlib import Path
from typing import Sequence, Union

import pytest
from _pytest.fixtures import SubRequest
from transformers import BertModel

from lightning_ir.bi_encoder.colbert import ColBERTModel, ColBERTModule
from lightning_ir.bi_encoder.model import BiEncoderConfig, BiEncoderModel
from lightning_ir.bi_encoder.module import BiEncoderModule
from lightning_ir.bi_encoder.xtr import XTRModel, XTRModule
from lightning_ir.data.datamodule import LightningIRDataModule
from lightning_ir.data.dataset import RunDataset, TupleDataset
from lightning_ir.loss.loss import (
    ConstantMarginMSE,
    InBatchCrossEntropy,
    LossFunction,
    RankNet,
    SupervisedMarginMSE,
)

DATA_DIR = Path(__file__).parent / "data"


class TestModel(BiEncoderModel):
    config_class = BiEncoderConfig

    def __init__(self, config: BiEncoderConfig) -> None:
        config.num_hidden_layers = 1
        super().__init__(config, "bert")
        self.bert = BertModel.from_pretrained(
            config.name_or_path, config=config, add_pooling_layer=False
        )


class TestModule(BiEncoderModule):
    def __init__(
        self,
        model: TestModel | None = None,
        model_name_or_path: str | None = None,
        config: BiEncoderConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, BiEncoderConfig):
                    raise ValueError(
                        "To initialize a new model pass a BiEncoderConfig."
                    )
                model = TestModel(config)
            else:
                model = TestModel.from_pretrained(model_name_or_path, config=config)
        super().__init__(model, loss_functions, evaluation_metrics)


MODULE_MAP = {
    TestModel: TestModule,
    ColBERTModel: ColBERTModule,
    XTRModel: XTRModule,
}
MODELS = Union[TestModel, ColBERTModel, XTRModel]
MODULES = Union[TestModule, ColBERTModule, XTRModule]


@pytest.fixture(scope="module", params=list(MODULE_MAP.keys()))
def model(model_name_or_path: str, request: SubRequest) -> MODELS:
    Model = request.param
    config = Model.config_class.from_pretrained(model_name_or_path, num_hidden_layers=1)
    _model = Model.from_pretrained(model_name_or_path, config=config)
    if config.add_marker_tokens:
        _model.encoder.resize_token_embeddings(_model.config.vocab_size + 2, 8)
    return _model


@pytest.fixture(
    scope="module",
    params=[
        ConstantMarginMSE(),
        RankNet(),
        SupervisedMarginMSE(),
        InBatchCrossEntropy("first", "first"),
    ],
)
def module(model: MODELS, request: SubRequest) -> MODULES:
    loss_function = request.param
    module = MODULE_MAP[type(model)](
        model, loss_functions=[loss_function], evaluation_metrics=["nDCG@10", "loss"]
    )
    return module


def tuples_datamodule(
    model: MODELS, inference_datasets: Sequence[RunDataset]
) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        model_name_or_path=model.config.name_or_path,
        config=model.config,
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
        train_dataset=TupleDataset("lightning-ir", targets="order", num_docs=2),
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


def test_doc_padding(model: MODELS, inference_datasets: Sequence[RunDataset]):
    datamodule = tuples_datamodule(model, inference_datasets)
    batch = next(iter(datamodule.train_dataloader()))
    doc_encoding = batch.doc_encoding
    doc_encoding["input_ids"] = doc_encoding["input_ids"][:-1]
    doc_encoding["attention_mask"] = doc_encoding["attention_mask"][:-1]
    doc_encoding["token_type_ids"] = doc_encoding["token_type_ids"][:-1]

    query_embedding = model.encode_queries(**batch.query_encoding)
    doc_embedding = model.encode_docs(**batch.doc_encoding)
    query_scoring_mask, doc_scoring_mask = model.scoring_masks(
        batch.query_encoding.input_ids,
        batch.doc_encoding.input_ids,
        batch.query_encoding.attention_mask,
        batch.doc_encoding.attention_mask,
    )
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            doc_embedding,
            query_scoring_mask,
            doc_scoring_mask,
            None,
        )
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            doc_embedding,
            query_scoring_mask,
            doc_scoring_mask,
            [doc_embedding.shape[0]],
        )
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            doc_embedding,
            query_scoring_mask,
            doc_scoring_mask,
            [0] * query_embedding.shape[0],
        )

    num_docs = [len(docs) for docs in batch.doc_ids]
    num_docs[-1] = num_docs[-1] - 1
    scores = model.score(
        query_embedding,
        doc_embedding,
        query_scoring_mask,
        doc_scoring_mask,
        num_docs,
    )
    assert scores.shape[0] == doc_embedding.shape[0]


def test_training_step(module: MODULES, inference_datasets: Sequence[RunDataset]):
    datamodule = tuples_datamodule(module.model, inference_datasets)
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = module.training_step(batch, 0)
    assert loss


def test_validation(module: MODULES, inference_datasets: Sequence[RunDataset]):
    datamodule = tuples_datamodule(module.model, inference_datasets)
    dataloader = datamodule.val_dataloader()[0]
    for batch, batch_idx in zip(dataloader, range(2)):
        module.validation_step(batch, batch_idx, 0)

    metrics = module.on_validation_epoch_end()
    assert metrics is not None
    for key, value in metrics.items():
        metric = key.split("/")[1]
        assert metric in {"nDCG@10"} or "validation" in metric
        assert value


def test_seralize_deserialize(
    model: TestModel | ColBERTModel | XTRModel, tmpdir_factory: pytest.TempdirFactory
):
    save_dir = tmpdir_factory.mktemp(model.config_class.model_type)
    model.save_pretrained(save_dir)
    kwargs = {}
    if isinstance(model, (ColBERTModel, XTRModel)):
        kwargs["mask_punctuation"] = False
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
        if key == "mask_punctuation":
            assert value and not getattr(new_model.config, key)
            continue
        assert getattr(new_model.config, key) == value
    for key, value in model.state_dict().items():
        assert new_model.state_dict()[key].equal(value)
