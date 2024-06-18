import os
from pathlib import Path
from typing import Sequence, List

import pytest
from _pytest.fixtures import SubRequest

from lightning_ir.bi_encoder.model import BiEncoderConfig
from lightning_ir.cross_encoder.model import CrossEncoderConfig
from lightning_ir.data.datamodule import LightningIRDataModule
from lightning_ir.data.dataset import DocDataset, QueryDataset, RunDataset, TupleDataset
from lightning_ir.data.ir_datasets_utils import register_local

DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
RUNS_DIR = DATA_DIR / "runs"

register_local(
    dataset_id="lightning-ir",
    docs=str(CORPUS_DIR / "docs.tsv"),
    queries=str(CORPUS_DIR / "queries.tsv"),
    qrels=str(CORPUS_DIR / "qrels.tsv"),
    docpairs=str(CORPUS_DIR / "docpairs.tsv"),
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def model_name_or_path():
    return "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def inference_datasets() -> List[RunDataset]:
    inference_datasets = [
        RunDataset(
            RUNS_DIR / run_path,
            depth=10,
            sample_size=10,
            sampling_strategy="top",
            targets="relevance",
        )
        for run_path in ["run.jsonl", "lightning-ir.tsv"]
    ]
    return inference_datasets


@pytest.fixture(scope="session", params=[BiEncoderConfig(), CrossEncoderConfig()])
def rank_run_datamodule(
    model_name_or_path: str,
    inference_datasets: Sequence[RunDataset],
    request: SubRequest,
) -> LightningIRDataModule:
    train_dataset = RunDataset(
        RUNS_DIR / "lightning-ir.tsv",
        depth=5,
        sample_size=2,
        sampling_strategy="top",
        targets="rank",
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
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
        RUNS_DIR / "lightning-ir.tsv",
        depth=5,
        sample_size=2,
        sampling_strategy="top",
        targets="relevance",
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
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
        RUNS_DIR / "lightning-ir.tsv",
        depth=5,
        sample_size=2,
        sampling_strategy="single_relevant",
        targets="relevance",
    )
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
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
    train_dataset = TupleDataset("lightning-ir", targets="order", num_docs=2)
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=request.param,
        num_workers=0,
        train_batch_size=2,
        inference_batch_size=2,
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
        inference_batch_size=2,
        inference_datasets=[QueryDataset("lightning-ir", num_queries=2)],
    )
    datamodule.setup(stage="predict")
    return datamodule


@pytest.fixture(scope="session")
def doc_datamodule(model_name_or_path: str) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        model_name_or_path=model_name_or_path,
        config=BiEncoderConfig(),
        num_workers=0,
        inference_batch_size=2,
        inference_datasets=[DocDataset("lightning-ir")],
    )
    datamodule.setup(stage="predict")
    return datamodule
