import torch

from lightning_ir.data.data import (
    BiEncoderRunBatch,
    CrossEncoderRunBatch,
    IndexBatch,
    SearchBatch,
)
from lightning_ir.data.datamodule import LightningIRDataModule
from lightning_ir.data.dataset import RunDataset, RunDatasetConfig

from .conftest import DATA_DIR


def test_rank_run_dataset(rank_run_datamodule: LightningIRDataModule):
    datamodule = rank_run_datamodule
    dataloader = datamodule.train_dataloader()
    config = datamodule.train_dataset_config
    assert config is not None

    batch = next(iter(dataloader))
    assert isinstance(batch, (BiEncoderRunBatch, CrossEncoderRunBatch))
    for value in batch:
        assert value is not None
    assert batch.targets.shape[0] == datamodule.train_batch_size * config.sample_size
    assert (
        batch.targets[..., 0]
        == torch.arange(1, config.sample_size + 1).repeat(datamodule.train_batch_size)
    ).all()


def test_relevance_run_dataset(relevance_run_datamodule: LightningIRDataModule):
    datamodule = relevance_run_datamodule
    dataloader = datamodule.train_dataloader()
    config = datamodule.train_dataset_config
    assert config is not None

    batch = next(iter(dataloader))
    assert isinstance(batch, (BiEncoderRunBatch, CrossEncoderRunBatch))
    for value in batch:
        assert value is not None
    assert batch.targets.shape[0] == datamodule.train_batch_size * config.sample_size


def test_single_relevant_run_dataset(
    single_relevant_run_datamodule: LightningIRDataModule,
):
    datamodule = single_relevant_run_datamodule
    dataloader = datamodule.train_dataloader()

    batch = next(iter(dataloader))
    assert isinstance(batch, (BiEncoderRunBatch, CrossEncoderRunBatch))
    for value in batch:
        assert value is not None
    assert (batch.targets > 0).sum() == datamodule.train_batch_size


def test_tuples_dataset(tuples_datamodule: LightningIRDataModule):
    dataloader = tuples_datamodule.train_dataloader()
    config = tuples_datamodule.train_dataset_config
    assert config is not None

    batch = next(iter(dataloader))
    assert isinstance(batch, (BiEncoderRunBatch, CrossEncoderRunBatch))
    for field in batch._fields:
        value = getattr(batch, field)
        if field == "qrels":
            assert value is None
        else:
            assert value is not None
    assert (
        batch.targets.shape[0]
        == dataloader.batch_size * dataloader.dataset.config.num_docs
    )


def test_query_dataset(query_datamodule: LightningIRDataModule):
    dataloader = query_datamodule.predict_dataloader()[0]
    batch: SearchBatch = next(iter(dataloader))
    assert isinstance(batch, SearchBatch)
    for field in batch._fields:
        value = getattr(batch, field)
        if field in ("query_encoding", "query_ids"):
            assert value is not None
        else:
            assert value is None


def test_doc_dataset(doc_datamodule: LightningIRDataModule):
    dataloader = doc_datamodule.predict_dataloader()[0]
    batch: IndexBatch = next(iter(dataloader))
    assert isinstance(batch, IndexBatch)
    for field in batch._fields:
        value = getattr(batch, field)
        if field in ("doc_encoding", "doc_ids"):
            assert value is not None
        else:
            assert value is None


def test_json_dataset():
    dataset = RunDataset(
        DATA_DIR / "run.jsonl",
        RunDatasetConfig("rank", depth=5, sample_size=5, sampling_strategy="top"),
    )
    sample = dataset[0]
    assert sample is not None
    assert sample.query_id is not None
    assert len(sample.doc_ids) == 5
    assert sample.qrels is None
