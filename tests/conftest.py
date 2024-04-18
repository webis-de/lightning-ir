import os
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest

from lightning_ir.bi_encoder.bi_encoder import BiEncoderConfig
from lightning_ir.cross_encoder.cross_encoder import CrossEncoderConfig
from lightning_ir.data.datamodule import (
    DocDatasetConfig,
    LightningIRDataModule,
    QueryDatasetConfig,
    RunDatasetConfig,
    TupleDatasetConfig,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def model_name_or_path():
    return "sentence-transformers/all-MiniLM-L6-v2"


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def rank_run_datamodule(
    model_name_or_path: str, request: SubRequest
) -> LightningIRDataModule:
    config = RunDatasetConfig(
        targets="rank", depth=10, sample_size=10, sampling_strategy="top"
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=str(DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run"),
        train_dataset_config=config,
        inference_datasets=[
            str(DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run"),
            str(DATA_DIR / "clueweb09-en-trec-web-2009-diversity.jsonl"),
        ],
        inference_dataset_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def relevance_run_datamodule(
    model_name_or_path: str, request: SubRequest
) -> LightningIRDataModule:
    config = RunDatasetConfig(
        targets="relevance", depth=10, sample_size=10, sampling_strategy="top"
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=str(DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run"),
        train_dataset_config=config,
        inference_datasets=[
            str(DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run"),
            str(DATA_DIR / "clueweb09-en-trec-web-2009-diversity.jsonl"),
        ],
        inference_dataset_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def single_relevant_run_datamodule(
    model_name_or_path: str, request
) -> LightningIRDataModule:
    config = RunDatasetConfig(
        targets="relevance",
        depth=200,
        sample_size=10,
        sampling_strategy="single_relevant",
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=str(DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run"),
        train_dataset_config=config,
        inference_datasets=[
            str(DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run"),
            str(DATA_DIR / "clueweb09-en-trec-web-2009-diversity.jsonl"),
        ],
        inference_dataset_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def tuples_datamodule(
    model_name_or_path: str, request: SubRequest
) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset="msmarco-passage/train/kd-docpairs",
        train_dataset_config=TupleDatasetConfig(2),
        inference_datasets=[
            str(DATA_DIR / "clueweb09-en-trec-web-2009-diversity.jsonl"),
            str(DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run"),
        ],
        inference_dataset_config=RunDatasetConfig(
            "relevance", depth=10, sample_size=10, sampling_strategy="top"
        ),
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session")
def query_datamodule(model_name_or_path: str) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=BiEncoderConfig(),
        num_workers=0,
        inference_batch_size=3,
        inference_datasets=["msmarco-passage/trec-dl-2019/judged"],
        inference_dataset_config=QueryDatasetConfig(),
    )
    datamodule.setup(stage="predict")
    return datamodule


@pytest.fixture(scope="session")
def doc_datamodule(model_name_or_path: str) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=BiEncoderConfig(),
        num_workers=0,
        inference_batch_size=3,
        inference_datasets=["msmarco-passage"],
        inference_dataset_config=DocDatasetConfig(num_docs=32),
    )
    datamodule.setup(stage="predict")
    return datamodule
