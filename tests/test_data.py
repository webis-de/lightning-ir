import torch
from tide.data import Batch
from tide.datamodule import DataModule, RunDatasetConfig


def test_rank_run_dataset(rank_run_datamodule: DataModule):
    datamodule = rank_run_datamodule
    dataloader = datamodule.train_dataloader()
    config = datamodule.train_run_config
    assert config is not None

    batch: Batch = next(iter(dataloader))
    assert batch.targets.shape[0] == datamodule.batch_size * config.sample_size
    assert (
        batch.targets
        == torch.arange(1, config.sample_size + 1).repeat(datamodule.batch_size)
    ).all()


def test_relevance_run_dataset(relevance_run_datamodule: DataModule):
    datamodule = relevance_run_datamodule
    dataloader = datamodule.train_dataloader()
    config = datamodule.train_run_config
    assert config is not None

    batch: Batch = next(iter(dataloader))
    assert batch.targets.shape[0] == datamodule.batch_size * config.sample_size
    assert (
        batch.targets
        != torch.arange(1, config.sample_size + 1).repeat(datamodule.batch_size)
    ).any()


def test_single_relevant_run_dataset(single_relevant_run_datamodule: DataModule):
    datamodule = single_relevant_run_datamodule
    dataloader = datamodule.train_dataloader()

    batch: Batch = next(iter(dataloader))
    assert (batch.targets > 0).sum() == datamodule.batch_size
    assert batch.relevance is None


def test_triples_dataset(triples_datamodule: DataModule):
    dataloader = triples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    assert batch.targets.shape[0] == triples_datamodule.batch_size * 2


def test_tokenizer(triples_datamodule: DataModule):
    triples_datamodule.config.query_expansion = True
    dataloader = triples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    assert (
        batch.query_encoding.input_ids[0, 1]
        == triples_datamodule.tokenizer.query_token_id
    )
    assert (
        batch.doc_encoding.input_ids[0, 1] == triples_datamodule.tokenizer.doc_token_id
    )
