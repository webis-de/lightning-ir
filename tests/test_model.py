from pathlib import Path
from typing import Sequence

import pytest
from transformers import AutoModel

from lightning_ir.base import LightningIRModel, LightningIRModule
from lightning_ir.cross_encoder import CrossEncoderModule
from lightning_ir.data import LightningIRDataModule, RunDataset, TupleDataset
from lightning_ir.loss.loss import InBatchLossFunction
from lightning_ir.main import LightningIRTrainer

DATA_DIR = Path(__file__).parent / "data"


def tuples_datamodule(module: LightningIRModule, inference_datasets: Sequence[RunDataset]) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=2,
        train_dataset=TupleDataset("lightning-ir", targets="order", num_docs=2),
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


def test_training_step(train_module: LightningIRModule, inference_datasets: Sequence[RunDataset]):
    if train_module.loss_functions is None:
        pytest.skip("Loss function is not set")
    datamodule = tuples_datamodule(train_module, inference_datasets)
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    if isinstance(train_module, CrossEncoderModule) and any(
        isinstance(loss_function[0], InBatchLossFunction) for loss_function in train_module.loss_functions
    ):
        with pytest.raises(RuntimeError):
            loss = train_module.training_step(batch, 0)
        return
    loss = train_module.training_step(batch, 0)
    assert loss


def test_validation(train_module: LightningIRModule, inference_datasets: Sequence[RunDataset]):
    datamodule = tuples_datamodule(train_module, inference_datasets)
    dataloader = datamodule.val_dataloader()[0]
    for batch, batch_idx in zip(dataloader, range(2)):
        train_module.validation_step(batch, batch_idx, 0)

    trainer = LightningIRTrainer(logger=False)
    trainer.validate(model=train_module, dataloaders=dataloader)

    metrics = trainer.callback_metrics
    assert metrics is not None
    for key, value in metrics.items():
        if not key:
            continue
        metric = key.split("/")[1]
        assert metric in {"nDCG@10"} or "validation" in metric
        assert value


def test_seralize_deserialize(module: LightningIRModule, tmp_path: Path):
    model = module.model
    save_dir = str(tmp_path / model.config_class.model_type)
    model.save_pretrained(save_dir)
    new_models = [
        model.__class__.from_pretrained(save_dir),
        model.__class__.__bases__[0].from_pretrained(save_dir),
        LightningIRModel.from_pretrained(save_dir),
        AutoModel.from_pretrained(save_dir),
    ]
    for new_model in new_models:
        for key, value in model.config.__dict__.items():
            if key in (
                "torch_dtype",
                "_name_or_path",
                "_commit_hash",
                "transformers_version",
                "_attn_implementation_autoset",
            ):
                continue
            assert getattr(new_model.config, key) == value
        for key, value in model.state_dict().items():
            assert new_model.state_dict()[key].equal(value)
