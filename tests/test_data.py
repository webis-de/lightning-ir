import torch

from mvr.data import IndexBatch, SearchBatch, TrainBatch
from mvr.datamodule import MVRDataModule


def test_rank_run_dataset(rank_run_datamodule: MVRDataModule):
    datamodule = rank_run_datamodule
    dataloader = datamodule.train_dataloader()
    config = datamodule.train_dataset_config
    assert config is not None

    batch: TrainBatch = next(iter(dataloader))
    assert isinstance(batch, TrainBatch)
    for value in batch:
        assert value is not None
    assert batch.targets.shape[0] == datamodule.train_batch_size * config.sample_size
    assert (
        batch.targets
        == torch.arange(1, config.sample_size + 1).repeat(datamodule.train_batch_size)
    ).all()


def test_relevance_run_dataset(relevance_run_datamodule: MVRDataModule):
    datamodule = relevance_run_datamodule
    dataloader = datamodule.train_dataloader()
    config = datamodule.train_dataset_config
    assert config is not None

    batch: TrainBatch = next(iter(dataloader))
    assert isinstance(batch, TrainBatch)
    for value in batch:
        assert value is not None
    assert batch.targets.shape[0] == datamodule.train_batch_size * config.sample_size
    assert (
        batch.targets
        != torch.arange(1, config.sample_size + 1).repeat(datamodule.train_batch_size)
    ).any()


def test_single_relevant_run_dataset(single_relevant_run_datamodule: MVRDataModule):
    datamodule = single_relevant_run_datamodule
    dataloader = datamodule.train_dataloader()

    batch: TrainBatch = next(iter(dataloader))
    assert isinstance(batch, TrainBatch)
    for value in batch:
        assert value is not None
    assert (batch.targets > 0).sum() == datamodule.train_batch_size


def test_tuples_dataset(tuples_datamodule: MVRDataModule):
    dataloader = tuples_datamodule.train_dataloader()
    config = tuples_datamodule.train_dataset_config
    assert config is not None

    batch: TrainBatch = next(iter(dataloader))
    assert isinstance(batch, TrainBatch)
    for field in batch._fields:
        value = getattr(batch, field)
        if field == "relevances":
            assert value is None
        else:
            assert value is not None
    assert (
        batch.targets.shape[0]
        == dataloader.batch_size * dataloader.dataset.config.num_docs
    )


def test_query_dataset(query_datamodule: MVRDataModule):
    dataloader = query_datamodule.predict_dataloader()[0]

    batch: SearchBatch = next(iter(dataloader))
    assert isinstance(batch, SearchBatch)
    for field in batch._fields:
        value = getattr(batch, field)
        if field in ("query_encoding", "query_ids"):
            assert value is not None
        else:
            assert value is None


def test_doc_dataset(doc_datamodule: MVRDataModule):
    dataloader = doc_datamodule.predict_dataloader()[0]
    batch: IndexBatch = next(iter(dataloader))

    assert isinstance(batch, IndexBatch)
    for field in batch._fields:
        value = getattr(batch, field)
        if field in ("doc_encoding", "doc_ids"):
            assert value is not None
        else:
            assert value is None


def test_tokenizer(tuples_datamodule: MVRDataModule):
    tuples_datamodule.config.query_expansion = True
    dataloader = tuples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    for field in batch._fields:
        value = getattr(batch, field)
        if field == "relevances":
            assert value is None
        else:
            assert value is not None
    assert (
        batch.query_encoding.input_ids[0, 1]
        == tuples_datamodule.tokenizer.query_token_id
    )
    assert (
        batch.doc_encoding.input_ids[0, 1] == tuples_datamodule.tokenizer.doc_token_id
    )
