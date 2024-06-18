import os
from pathlib import Path
from typing import List

import pytest

from lightning_ir.data.dataset import RunDataset
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
