from typing import Sequence

import pytest
import torch

from lightning_ir.data.data import IndexBatch, SearchBatch, TrainBatch
from lightning_ir.data.datamodule import LightningIRDataModule
from lightning_ir.data.dataset import RunDataset, TupleDataset

from .conftest import RUNS_DIR


@pytest.fixture()
def rank_run_datamodule(inference_datasets: Sequence[RunDataset]) -> LightningIRDataModule:
    train_dataset = RunDataset(
        RUNS_DIR / "lightning-ir.tsv",
        depth=5,
        sample_size=2,
        sampling_strategy="top",
        targets="rank",
    )
    datamodule = LightningIRDataModule(
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
        train_dataset=train_dataset,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture()
def relevance_run_datamodule(inference_datasets: Sequence[RunDataset]) -> LightningIRDataModule:
    train_dataset = RunDataset(
        RUNS_DIR / "lightning-ir.tsv",
        depth=5,
        sample_size=2,
        sampling_strategy="top",
        targets="relevance",
    )
    datamodule = LightningIRDataModule(
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
        train_dataset=train_dataset,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture()
def single_relevant_run_datamodule(inference_datasets: Sequence[RunDataset]) -> LightningIRDataModule:
    train_dataset = RunDataset(
        RUNS_DIR / "lightning-ir.tsv",
        depth=5,
        sample_size=2,
        sampling_strategy="single_relevant",
        targets="relevance",
    )
    datamodule = LightningIRDataModule(
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
        train_dataset=train_dataset,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture()
def tuples_datamodule(inference_datasets: Sequence[RunDataset]) -> LightningIRDataModule:
    train_dataset = TupleDataset("lightning-ir", targets="order", num_docs=2)
    datamodule = LightningIRDataModule(
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
        train_dataset=train_dataset,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


def test_rank_run_dataset(rank_run_datamodule: LightningIRDataModule):
    datamodule = rank_run_datamodule
    dataloader = datamodule.train_dataloader()
    dataset = dataloader.dataset

    batch = next(iter(dataloader))
    assert isinstance(batch, TrainBatch)
    for value in batch.__dict__.values():
        assert value is not None
    assert batch.targets.shape[0] == datamodule.train_batch_size * dataset.sample_size
    target_ranks = dataset.depth - torch.arange(dataset.sample_size).repeat(datamodule.train_batch_size)
    assert (batch.targets[..., 0] == target_ranks).all()


def test_relevance_run_dataset(relevance_run_datamodule: LightningIRDataModule):
    datamodule = relevance_run_datamodule
    dataloader = datamodule.train_dataloader()
    dataset = dataloader.dataset

    batch = next(iter(dataloader))
    assert isinstance(batch, TrainBatch)
    for value in batch.__dict__.values():
        assert value is not None
    assert batch.targets.shape[0] == datamodule.train_batch_size * dataset.sample_size


def test_single_relevant_run_dataset(
    single_relevant_run_datamodule: LightningIRDataModule,
):
    datamodule = single_relevant_run_datamodule
    dataloader = datamodule.train_dataloader()

    batch = next(iter(dataloader))
    assert isinstance(batch, TrainBatch)
    for value in batch.__dict__.values():
        assert value is not None
    assert (batch.targets.max(-1).values > 0).sum() == datamodule.train_batch_size


def test_tuples_dataset(tuples_datamodule: LightningIRDataModule):
    dataloader = tuples_datamodule.train_dataloader()
    dataset = dataloader.dataset

    batch = next(iter(dataloader))
    assert isinstance(batch, TrainBatch)
    for field in batch.__dict__.keys():
        value = getattr(batch, field)
        if field == "qrels":
            assert value is None
        else:
            assert value is not None
    assert batch.targets.shape[0] == dataloader.batch_size * dataset.num_docs


def test_query_dataset(query_datamodule: LightningIRDataModule):
    dataloader = query_datamodule.test_dataloader()[0]
    batch: SearchBatch = next(iter(dataloader))
    assert isinstance(batch, SearchBatch)
    for field in batch.__dict__.keys():
        value = getattr(batch, field)
        if field in ("queries", "query_ids", "qrels"):
            assert value is not None
        else:
            assert value is None


def test_doc_dataset(doc_datamodule: LightningIRDataModule):
    dataloader = doc_datamodule.test_dataloader()[0]
    batch: IndexBatch = next(iter(dataloader))
    assert isinstance(batch, IndexBatch)
    for field in batch.__dict__.keys():
        value = getattr(batch, field)
        if field in ("docs", "doc_ids"):
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
    )
    sample = dataset[0]
    assert sample is not None
    assert sample.query_id is not None
    assert len(sample.doc_ids) == 5
