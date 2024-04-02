import os
from pathlib import Path

import pytest

from mvr.datamodule import (
    DocDatasetConfig,
    MVRDataModule,
    QueryDatasetConfig,
    RunDatasetConfig,
    TupleDatasetConfig,
)
from mvr.mvr import MVRConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=str(
            Path(__file__).parent / "data" / "msmarco-passage-trec-dl-2019-judged.run"
        ),
        train_dataset_config=config,
        inference_datasets=[
            str(
                Path(__file__).parent
                / "data"
                / "msmarco-passage-trec-dl-2019-judged.run"
            )
        ],
        inference_dataset_config=RunDatasetConfig(
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
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=str(
            Path(__file__).parent / "data" / "msmarco-passage-trec-dl-2019-judged.run"
        ),
        train_dataset_config=config,
        inference_datasets=[
            str(
                Path(__file__).parent
                / "data"
                / "msmarco-passage-trec-dl-2019-judged.run"
            )
        ],
        inference_dataset_config=RunDatasetConfig(
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
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=str(
            Path(__file__).parent / "data" / "msmarco-passage-trec-dl-2019-judged.run"
        ),
        train_dataset_config=config,
        inference_datasets=[
            str(
                Path(__file__).parent
                / "data"
                / "msmarco-passage-trec-dl-2019-judged.run"
            )
        ],
        inference_dataset_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session")
def tuples_datamodule(model_name_or_path: str) -> MVRDataModule:
    datamodule = MVRDataModule(
        model_name_or_path=model_name_or_path,
        config=MVRConfig(),
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset="msmarco-passage/train/kd-docpairs",
        train_dataset_config=TupleDatasetConfig(2),
        inference_datasets=["msmarco-passage/train/kd-docpairs"],
        inference_dataset_config=TupleDatasetConfig(2),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session")
def query_datamodule(model_name_or_path: str) -> MVRDataModule:
    datamodule = MVRDataModule(
        model_name_or_path=model_name_or_path,
        config=MVRConfig(),
        num_workers=0,
        inference_batch_size=3,
        inference_datasets=["msmarco-passage/trec-dl-2019/judged"],
        inference_dataset_config=QueryDatasetConfig(),
    )
    datamodule.setup(stage="predict")
    return datamodule


@pytest.fixture(scope="session")
def doc_datamodule(model_name_or_path: str) -> MVRDataModule:
    datamodule = MVRDataModule(
        model_name_or_path=model_name_or_path,
        config=MVRConfig(),
        num_workers=0,
        inference_batch_size=3,
        inference_datasets=["msmarco-passage", "msmarco-passage"],
        inference_dataset_config=DocDatasetConfig(num_docs=32),
    )
    datamodule.setup(stage="predict")
    return datamodule
