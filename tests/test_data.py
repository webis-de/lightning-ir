import torch

from lightning_ir.data.data import (
    BiEncoderRunBatch,
    CrossEncoderRunBatch,
    IndexBatch,
    SearchBatch,
)
from lightning_ir.data.datamodule import LightningIRDataModule
from lightning_ir.data.dataset import RunDataset

from .conftest import RUNS_DIR


def test_rank_run_dataset(rank_run_datamodule: LightningIRDataModule):
    datamodule = rank_run_datamodule
    dataloader = datamodule.train_dataloader()
    dataset = dataloader.dataset

    batch = next(iter(dataloader))
    assert isinstance(batch, (BiEncoderRunBatch, CrossEncoderRunBatch))
    for value in batch:
        assert value is not None
    assert batch.targets.shape[0] == datamodule.train_batch_size * dataset.sample_size
    target_ranks = dataset.depth - torch.arange(dataset.sample_size).repeat(
        datamodule.train_batch_size
    )
    assert (batch.targets[..., 0] == target_ranks).all()


def test_relevance_run_dataset(relevance_run_datamodule: LightningIRDataModule):
    datamodule = relevance_run_datamodule
    dataloader = datamodule.train_dataloader()
    dataset = dataloader.dataset

    batch = next(iter(dataloader))
    assert isinstance(batch, (BiEncoderRunBatch, CrossEncoderRunBatch))
    for value in batch:
        assert value is not None
    assert batch.targets.shape[0] == datamodule.train_batch_size * dataset.sample_size


def test_single_relevant_run_dataset(
    single_relevant_run_datamodule: LightningIRDataModule,
):
    datamodule = single_relevant_run_datamodule
    dataloader = datamodule.train_dataloader()

    batch = next(iter(dataloader))
    assert isinstance(batch, (BiEncoderRunBatch, CrossEncoderRunBatch))
    for value in batch:
        assert value is not None
    assert (batch.targets.max(-1).values > 0).sum() == datamodule.train_batch_size


def test_tuples_dataset(tuples_datamodule: LightningIRDataModule):
    dataloader = tuples_datamodule.train_dataloader()
    dataset = dataloader.dataset

    batch = next(iter(dataloader))
    assert isinstance(batch, (BiEncoderRunBatch, CrossEncoderRunBatch))
    for field in batch._fields:
        value = getattr(batch, field)
        if field == "qrels":
            assert value is None
        else:
            assert value is not None
    assert batch.targets.shape[0] == dataloader.batch_size * dataset.num_docs


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
        RUNS_DIR / "run.jsonl",
        depth=5,
        sample_size=5,
        sampling_strategy="top",
        targets="rank",
    ).setup()
    sample = dataset[0]
    assert sample is not None
    assert sample.query_id is not None
    assert len(sample.doc_ids) == 5
