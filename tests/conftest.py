from pathlib import Path

import pytest

from tide.datamodule import MVRDataModule, RunDatasetConfig
from tide.mvr import MVRConfig


@pytest.fixture(scope="session")
def model_name_or_path():
    return "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def rank_run_datamodule(model_name_or_path: str) -> MVRDataModule:
    config = RunDatasetConfig(
        targets="rank", depth=10, sample_size=10, sampling_strategy="top"
    )
    datamodule = MVRDataModule(
        model_name_or_path=model_name_or_path,
        config=MVRConfig(),
        num_workers=0,
        batch_size=3,
        train_dataset=str(
            Path(__file__).parent / "data" / "msmarco-passage-trec-dl-2019-judged.run"
        ),
        train_run_config=config,
        inference_datasets=[
            str(
                Path(__file__).parent
                / "data"
                / "msmarco-passage-trec-dl-2019-judged.run"
            )
        ],
        inference_run_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session")
def relevance_run_datamodule(model_name_or_path: str) -> MVRDataModule:
    config = RunDatasetConfig(
        targets="relevance", depth=10, sample_size=10, sampling_strategy="top"
    )
    datamodule = MVRDataModule(
        model_name_or_path=model_name_or_path,
        config=MVRConfig(),
        num_workers=0,
        batch_size=3,
        train_dataset=str(
            Path(__file__).parent / "data" / "msmarco-passage-trec-dl-2019-judged.run"
        ),
        train_run_config=config,
        inference_datasets=[
            str(
                Path(__file__).parent
                / "data"
                / "msmarco-passage-trec-dl-2019-judged.run"
            )
        ],
        inference_run_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session")
def single_relevant_run_datamodule(model_name_or_path: str) -> MVRDataModule:
    config = RunDatasetConfig(
        targets="relevance",
        depth=200,
        sample_size=10,
        sampling_strategy="single_relevant",
    )
    datamodule = MVRDataModule(
        model_name_or_path=model_name_or_path,
        config=MVRConfig(),
        num_workers=0,
        batch_size=3,
        train_dataset=str(
            Path(__file__).parent / "data" / "msmarco-passage-trec-dl-2019-judged.run"
        ),
        train_run_config=config,
        inference_datasets=[
            str(
                Path(__file__).parent
                / "data"
                / "msmarco-passage-trec-dl-2019-judged.run"
            )
        ],
        inference_run_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session")
def triples_datamodule(model_name_or_path: str) -> MVRDataModule:
    model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    datamodule = MVRDataModule(
        model_name_or_path=model_name_or_path,
        config=MVRConfig(),
        num_workers=0,
        batch_size=3,
        train_dataset="msmarco-passage/train/kd-docpairs",
    )
    datamodule.setup(stage="fit")
    return datamodule
