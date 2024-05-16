import os
from pathlib import Path
from typing import Sequence, List

import pytest
from _pytest.fixtures import SubRequest

from lightning_ir.bi_encoder.model import BiEncoderConfig
from lightning_ir.cross_encoder.model import CrossEncoderConfig
from lightning_ir.data.datamodule import LightningIRDataModule
from lightning_ir.data.dataset import DocDataset, QueryDataset, RunDataset, TupleDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def model_name_or_path():
    return "sentence-transformers/all-MiniLM-L6-v2"


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def inference_datasets() -> List[RunDataset]:
    inference_datasets = [
        RunDataset(
            run_path,
            depth=10,
            sample_size=10,
            sampling_strategy="top",
            targets="relevance",
        )
        for run_path in [
            DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run",
            DATA_DIR / "clueweb09-en-trec-web-2009-diversity.jsonl",
        ]
    ]
    return inference_datasets


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def rank_run_datamodule(
    model_name_or_path: str,
    inference_datasets: Sequence[RunDataset],
    request: SubRequest,
) -> LightningIRDataModule:
    train_dataset = RunDataset(
        DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run",
        depth=10,
        sample_size=10,
        sampling_strategy="top",
        targets="rank",
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=train_dataset,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def relevance_run_datamodule(
    model_name_or_path: str,
    inference_datasets: Sequence[RunDataset],
    request: SubRequest,
) -> LightningIRDataModule:
    train_dataset = RunDataset(
        DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run",
        depth=10,
        sample_size=10,
        sampling_strategy="top",
        targets="relevance",
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=train_dataset,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def single_relevant_run_datamodule(
    model_name_or_path: str,
    inference_datasets: Sequence[RunDataset],
    request: SubRequest,
) -> LightningIRDataModule:
    train_dataset = RunDataset(
        DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run",
        depth=10,
        sample_size=10,
        sampling_strategy="single_relevant",
        targets="relevance",
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=train_dataset,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def tuples_datamodule(
    model_name_or_path: str,
    inference_datasets: Sequence[RunDataset],
    request: SubRequest,
) -> LightningIRDataModule:
    train_dataset = TupleDataset(
        "msmarco-passage/train/kd-docpairs", targets="score", num_docs=2
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=train_dataset,
        inference_datasets=inference_datasets,
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
        inference_datasets=[QueryDataset("msmarco-passage/trec-dl-2019/judged")],
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
        inference_datasets=[DocDataset("msmarco-passage", num_docs=32)],
    )
    datamodule.setup(stage="predict")
    return datamodule
