import os
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any, Union

import pytest
from _pytest.fixtures import SubRequest

from lightning_ir import (
    BiEncoderConfig,
    BiEncoderModule,
    CrossEncoderConfig,
    CrossEncoderModule,
    DocDataset,
    InBatchCrossEntropy,
    LightningIRDataModule,
    LightningIRModule,
    QueryDataset,
    RankNet,
    RunDataset,
)
from lightning_ir.data.external_datasets.ir_datasets_utils import register_new_dataset
from lightning_ir.models import (
    CoilConfig,
    ColConfig,
    DprConfig,
    MonoConfig,
    MvrConfig,
    SetEncoderConfig,
    SpladeConfig,
)


def pytest_addoption(parser):
    parser.addoption(
        "--delete-cache",
        action="store_true",
        default=False,
        help="remove Hugging Face models after tests",
    )
    parser.addoption(
        "--run-models",
        action="store_true",
        default=False,
        help="run tests that require downloading models from Hugging Face",
    )
    parser.addoption(
        "--run-datasets",
        action="store_true",
        default=False,
        help="run tests that require downloading datasets from external sources",
    )


def pytest_collection_modifyitems(config, items):
    run_models = config.getoption("--run-models")
    run_datasets = config.getoption("--run-datasets")
    skip_model = pytest.mark.skip(reason="need --run-models option to run")
    skip_dataset = pytest.mark.skip(reason="need --run-datasets option to run")
    for item in items:
        if "model" in item.keywords and not run_models:
            item.add_marker(skip_model)
        if "dataset" in item.keywords and not run_datasets:
            item.add_marker(skip_dataset)


DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
RUNS_DIR = DATA_DIR / "runs"

CONFIGS = Union[BiEncoderConfig, CrossEncoderConfig]

register_new_dataset(
    dataset_id="lightning-ir",
    docs=str(CORPUS_DIR / "docs.tsv"),
    queries=str(CORPUS_DIR / "queries.tsv"),
    qrels=str(CORPUS_DIR / "qrels.tsv"),
    docpairs=str(CORPUS_DIR / "docpairs.tsv"),
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def model_name_or_path() -> str:
    return str(DATA_DIR / "test_model")


@pytest.fixture()
def inference_datasets() -> list[RunDataset]:
    inference_datasets = [
        RunDataset(RUNS_DIR / run_path, depth=10, sampling_strategy="top", targets="relevance")
        for run_path in ["run.jsonl", "lightning-ir.tsv"]
    ]
    return inference_datasets


DATA_DIR = Path(__file__).parent / "data"

GLOBAL_KWARGS: dict[str, Any] = {"query_length": 8, "doc_length": 8}

BI_ENCODER_GLOBAL_KWARGS: dict[str, Any] = {"embedding_dim": 4}

BI_ENCODER_CONFIGS = {
    "CoilModel": CoilConfig(**GLOBAL_KWARGS, **BI_ENCODER_GLOBAL_KWARGS),
    "ColModel": ColConfig(**GLOBAL_KWARGS, **BI_ENCODER_GLOBAL_KWARGS),
    "DprModel": DprConfig(**GLOBAL_KWARGS, **BI_ENCODER_GLOBAL_KWARGS),
    "MvrModel": MvrConfig(**GLOBAL_KWARGS, **BI_ENCODER_GLOBAL_KWARGS),
    "SpladeModel": SpladeConfig(**GLOBAL_KWARGS),
}


CROSS_ENCODER_CONFIGS = {
    "Mono": MonoConfig(**GLOBAL_KWARGS),
    "SetEncoder": SetEncoderConfig(**GLOBAL_KWARGS),
}

ALL_CONFIGS = {**BI_ENCODER_CONFIGS, **CROSS_ENCODER_CONFIGS}


@pytest.fixture(scope="module", params=list(ALL_CONFIGS.values()), ids=list(ALL_CONFIGS.keys()))
def config(request: SubRequest) -> CONFIGS:
    return request.param


@pytest.fixture(
    scope="module",
    params=list(BI_ENCODER_CONFIGS.values()),
    ids=list(BI_ENCODER_CONFIGS.keys()),
)
def bi_encoder_config(request: SubRequest) -> BiEncoderConfig:
    return request.param


@pytest.fixture(
    scope="module",
    params=list(CROSS_ENCODER_CONFIGS.values()),
    ids=list(CROSS_ENCODER_CONFIGS.keys()),
)
def cross_encoder_config(request: SubRequest) -> CrossEncoderConfig:
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        RankNet(),
        InBatchCrossEntropy("first", "first"),
    ],
    ids=["RankNet", "InBatchCrossEntropy"],
)
def train_module(config: CONFIGS, model_name_or_path: str, request: SubRequest) -> LightningIRModule:
    loss_function = request.param
    kwargs: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "config": config,
        "loss_functions": [loss_function],
        "evaluation_metrics": ["loss", "nDCG@10"],
    }
    module: LightningIRModule
    if isinstance(config, CrossEncoderConfig):
        module = CrossEncoderModule(**kwargs)
    elif isinstance(config, BiEncoderConfig):
        module = BiEncoderModule(**kwargs)
    else:
        raise ValueError(f"Invalid config type: {config}")
    return module


@pytest.fixture(scope="module")
def module(config: CONFIGS, model_name_or_path: str) -> LightningIRModule:
    kwargs: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "config": config,
        "evaluation_metrics": ["loss", "nDCG@10", "MRR@10"],
    }
    module: LightningIRModule
    if isinstance(config, CrossEncoderConfig):
        module = CrossEncoderModule(**kwargs)
    elif isinstance(config, BiEncoderConfig):
        module = BiEncoderModule(**kwargs)
    else:
        raise ValueError(f"Invalid config type: {config}")
    return module


@pytest.fixture(scope="module")
def bi_encoder_module(bi_encoder_config: BiEncoderConfig, model_name_or_path: str) -> LightningIRModule:
    kwargs: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "config": bi_encoder_config,
        "evaluation_metrics": ["loss", "nDCG@10"],
    }
    module = BiEncoderModule(**kwargs)
    return module


@pytest.fixture(scope="module")
def cross_encoder_module(cross_encoder_config: CrossEncoderConfig, model_name_or_path: str) -> LightningIRModule:
    kwargs: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "config": cross_encoder_config,
        "evaluation_metrics": ["loss", "nDCG@10"],
    }
    module = CrossEncoderModule(**kwargs)
    return module


@pytest.fixture()
def query_datamodule() -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        num_workers=0,
        inference_batch_size=2,
        inference_datasets=[QueryDataset("lightning-ir", num_queries=2)],
    )
    datamodule.setup(stage="test")
    return datamodule


@pytest.fixture()
def doc_datamodule() -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        num_workers=0,
        inference_batch_size=2,
        inference_datasets=[DocDataset("lightning-ir")],
    )
    datamodule.setup(stage="test")
    return datamodule


@pytest.fixture()
def hf_model(request: SubRequest) -> Generator[str, None, None]:
    yield request.param
    if request.config.getoption("--delete-cache"):
        hf_cache = os.environ.get("HF_HOME", os.path.join(Path.home(), ".cache/huggingface"))
        shutil.rmtree(hf_cache, ignore_errors=True)


@pytest.fixture()
def ir_datasets_run_path_or_id(request: SubRequest) -> Generator[str, None, None]:
    yield request.param
    if request.config.getoption("--delete-cache"):
        ir_datasets_cache = os.environ.get("IR_DATASETS_HOME", os.path.join(Path.home(), ".ir_datasets"))
        shutil.rmtree(ir_datasets_cache, ignore_errors=True)
