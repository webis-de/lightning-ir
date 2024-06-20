from pathlib import Path
from typing import Sequence

import pytest

from lightning_ir.base import LightningIRModule
from lightning_ir.bi_encoder import BiEncoderModule
from lightning_ir.cross_encoder import CrossEncoderModule
from lightning_ir.data import LightningIRDataModule, RunDataset, TupleDataset
from lightning_ir.loss.loss import InBatchLossFunction

DATA_DIR = Path(__file__).parent / "data"


def tuples_datamodule(
    module: LightningIRModule, inference_datasets: Sequence[RunDataset]
) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        module=module,
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
        train_dataset=TupleDataset("lightning-ir", targets="order", num_docs=2),
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


def test_doc_padding(
    bi_encoder_module: BiEncoderModule, inference_datasets: Sequence[RunDataset]
):
    datamodule = tuples_datamodule(bi_encoder_module, inference_datasets)
    model = bi_encoder_module.model
    batch = next(iter(datamodule.train_dataloader()))
    doc_encoding = batch.doc_encoding
    doc_encoding["input_ids"] = doc_encoding["input_ids"][:-1]
    doc_encoding["attention_mask"] = doc_encoding["attention_mask"][:-1]
    doc_encoding["token_type_ids"] = doc_encoding["token_type_ids"][:-1]

    query_embeddings = model.encode_query(**batch.query_encoding)
    doc_embeddings = model.encode_doc(**batch.doc_encoding)
    with pytest.raises(ValueError):
        model.score(query_embeddings, doc_embeddings)
    with pytest.raises(ValueError):
        model.score(
            query_embeddings, doc_embeddings, [doc_embeddings.embeddings.shape[0]]
        )
    with pytest.raises(ValueError):
        model.score(
            query_embeddings, doc_embeddings, [0] * query_embeddings.embeddings.shape[0]
        )

    num_docs = [len(docs) for docs in batch.doc_ids]
    num_docs[-1] = num_docs[-1] - 1
    scores = model.score(query_embeddings, doc_embeddings, num_docs)
    assert scores.shape[0] == doc_embeddings.embeddings.shape[0]


def test_training_step(
    train_module: LightningIRModule, inference_datasets: Sequence[RunDataset]
):
    if train_module.loss_functions is None:
        pytest.skip("Loss function is not set")
    datamodule = tuples_datamodule(train_module, inference_datasets)
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    if isinstance(train_module, CrossEncoderModule) and any(
        isinstance(loss_function, InBatchLossFunction)
        for loss_function in train_module.loss_functions
    ):
        with pytest.raises(RuntimeError):
            loss = train_module.training_step(batch, 0)
        return
    loss = train_module.training_step(batch, 0)
    assert loss


def test_validation(
    train_module: LightningIRModule, inference_datasets: Sequence[RunDataset]
):
    datamodule = tuples_datamodule(train_module, inference_datasets)
    dataloader = datamodule.val_dataloader()[0]
    for batch, batch_idx in zip(dataloader, range(2)):
        train_module.validation_step(batch, batch_idx, 0)

    metrics = train_module.on_validation_epoch_end()
    assert metrics is not None
    for key, value in metrics.items():
        metric = key.split("/")[1]
        assert metric in {"nDCG@10"} or "validation" in metric
        assert value


def test_seralize_deserialize(
    module: LightningIRModule, tmpdir_factory: pytest.TempdirFactory
):
    model = module.model
    save_dir = tmpdir_factory.mktemp(model.config_class.model_type)
    model.save_pretrained(save_dir)
    kwargs = {}
    new_models = [
        model.__class__.from_pretrained(save_dir, **kwargs),
        model.__class__.__bases__[0].from_pretrained(save_dir, **kwargs),
    ]
    for new_model in new_models:
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
