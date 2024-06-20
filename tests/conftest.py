import os
from pathlib import Path
from typing import List, Union

import pytest
from _pytest.fixtures import SubRequest

from lightning_ir import (
    BiEncoderConfig,
    BiEncoderModule,
    ConstantMarginMSE,
    CrossEncoderConfig,
    CrossEncoderModule,
    DocDataset,
    InBatchCrossEntropy,
    LightningIRDataModule,
    LightningIRModule,
    MultiVectorBiEncoderConfig,
    QueryDataset,
    RankNet,
    RunDataset,
    SingleVectorBiEncoderConfig,
    SupervisedMarginMSE,
)
from lightning_ir.data.ir_datasets_utils import register_local

DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
RUNS_DIR = DATA_DIR / "runs"

CONFIGS = Union[
    SingleVectorBiEncoderConfig, MultiVectorBiEncoderConfig, CrossEncoderConfig
]

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


DATA_DIR = Path(__file__).parent / "data"

BI_ENCODER_CONFIGS = [
    SingleVectorBiEncoderConfig(query_length=4, doc_length=8),
    MultiVectorBiEncoderConfig(query_length=4, doc_length=8),
]

CROSS_ENCODER_CONFIGS = [
    CrossEncoderConfig(query_length=4, doc_length=8),
]

ALL_CONFIGS = BI_ENCODER_CONFIGS + CROSS_ENCODER_CONFIGS


@pytest.fixture(scope="module", params=ALL_CONFIGS)
def config(request: SubRequest) -> CONFIGS:
    return request.param


@pytest.fixture(scope="module", params=BI_ENCODER_CONFIGS)
def bi_encoder_config(request: SubRequest) -> CONFIGS:
    return request.param


@pytest.fixture(scope="module", params=CROSS_ENCODER_CONFIGS)
def cross_encoder_config(request: SubRequest) -> CONFIGS:
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        RankNet(),
        InBatchCrossEntropy("first", "first"),
    ],
)
def train_module(
    config: CONFIGS, model_name_or_path: str, request: SubRequest
) -> LightningIRModule:
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
def bi_encoder_module(
    bi_encoder_config: BiEncoderConfig, model_name_or_path: str
) -> LightningIRModule:
    kwargs = dict(
        model_name_or_path=model_name_or_path,
        config=bi_encoder_config,
        evaluation_metrics=["loss", "nDCG@10"],
    )
    module = BiEncoderModule(**kwargs)
    return module


@pytest.fixture(scope="module")
def cross_encoder_module(
    cross_encoder_config: CrossEncoderConfig, model_name_or_path: str
) -> LightningIRModule:
    kwargs = dict(
        model_name_or_path=model_name_or_path,
        config=cross_encoder_config,
        evaluation_metrics=["loss", "nDCG@10"],
    )
    module = CrossEncoderModule(**kwargs)
    return module


@pytest.fixture()
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


@pytest.fixture()
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
