import os
from pathlib import Path
from typing import Any, Dict, List, Union

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
from lightning_ir.data.ir_datasets_utils import register_local

DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
RUNS_DIR = DATA_DIR / "runs"

CONFIGS = Union[BiEncoderConfig, CrossEncoderConfig]

register_local(
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
def inference_datasets() -> List[RunDataset]:
    inference_datasets = [
        RunDataset(RUNS_DIR / run_path, depth=10, sampling_strategy="top", targets="relevance")
        for run_path in ["run.jsonl", "lightning-ir.tsv"]
    ]
    return inference_datasets


DATA_DIR = Path(__file__).parent / "data"

GLOBAL_KWARGS: Dict[str, Any] = dict(query_length=8, doc_length=8)

BI_ENCODER_GLOBAL_KWARGS = dict(embedding_dim=4)

BI_ENCODER_CONFIGS = {
    "MultiVectorBiEncoder": BiEncoderConfig(
        query_pooling_strategy=None,
        doc_pooling_strategy=None,
        **GLOBAL_KWARGS,
        **BI_ENCODER_GLOBAL_KWARGS,
    ),
    "SingleVectorBiEncoder": BiEncoderConfig(**GLOBAL_KWARGS, **BI_ENCODER_GLOBAL_KWARGS),
}


CROSS_ENCODER_CONFIGS = {"CrossEncoder": CrossEncoderConfig(**GLOBAL_KWARGS)}

ALL_CONFIGS = {**BI_ENCODER_CONFIGS, **CROSS_ENCODER_CONFIGS}


@pytest.fixture(scope="module", params=list(ALL_CONFIGS.values()), ids=list(ALL_CONFIGS.keys()))
def config(request: SubRequest) -> CONFIGS:
    return request.param


@pytest.fixture(
    scope="module",
    params=list(BI_ENCODER_CONFIGS.values()),
    ids=list(BI_ENCODER_CONFIGS.keys()),
)
def bi_encoder_config(request: SubRequest) -> CONFIGS:
    return request.param


@pytest.fixture(
    scope="module",
    params=list(CROSS_ENCODER_CONFIGS.values()),
    ids=list(CROSS_ENCODER_CONFIGS.keys()),
)
def cross_encoder_config(request: SubRequest) -> CONFIGS:
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
    kwargs = dict(
        model_name_or_path=model_name_or_path,
        config=config,
        loss_functions=[loss_function],
        evaluation_metrics=["loss", "nDCG@10"],
    )
    if isinstance(config, CrossEncoderConfig):
        module = CrossEncoderModule(**kwargs)
    elif isinstance(config, BiEncoderConfig):
        module = BiEncoderModule(**kwargs)
    else:
        raise ValueError(f"Invalid config type: {config}")
    return module


@pytest.fixture(scope="module")
def module(config: CONFIGS, model_name_or_path: str) -> LightningIRModule:
    kwargs = dict(
        model_name_or_path=model_name_or_path,
        config=config,
        evaluation_metrics=["loss", "nDCG@10", "MRR@10"],
    )
    if isinstance(config, CrossEncoderConfig):
        module = CrossEncoderModule(**kwargs)
    elif isinstance(config, BiEncoderConfig):
        module = BiEncoderModule(**kwargs)
    else:
        raise ValueError(f"Invalid config type: {config}")
    return module


@pytest.fixture(scope="module")
def bi_encoder_module(bi_encoder_config: BiEncoderConfig, model_name_or_path: str) -> LightningIRModule:
    kwargs = dict(
        model_name_or_path=model_name_or_path,
        config=bi_encoder_config,
        evaluation_metrics=["loss", "nDCG@10"],
    )
    module = BiEncoderModule(**kwargs)
    return module


@pytest.fixture(scope="module")
def cross_encoder_module(cross_encoder_config: CrossEncoderConfig, model_name_or_path: str) -> LightningIRModule:
    kwargs = dict(
        model_name_or_path=model_name_or_path,
        config=cross_encoder_config,
        evaluation_metrics=["loss", "nDCG@10"],
    )
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
