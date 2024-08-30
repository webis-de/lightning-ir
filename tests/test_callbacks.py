from pathlib import Path
from typing import Sequence

import pandas as pd
import pytest
from _pytest.fixtures import SubRequest

from lightning_ir import BiEncoderModule, LightningIRDataModule, LightningIRModule, LightningIRTrainer, RunDataset
from lightning_ir.lightning_utils.callbacks import IndexCallback, ReRankCallback, SearchCallback
from lightning_ir.retrieve import (
    FaissFlatIndexConfig,
    FaissSearchConfig,
    SearchConfig,
    SparseIndexConfig,
    SparseSearchConfig,
)
from lightning_ir.retrieve.indexer import IndexConfig

from .conftest import DATA_DIR


@pytest.fixture(
    params=[FaissFlatIndexConfig(), SparseIndexConfig()],
    ids=["Faiss", "Sparse"],
)
def index_config(request: SubRequest) -> IndexConfig:
    return request.param


def run_datamodule(module: LightningIRModule, inference_datasets: Sequence[RunDataset]) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        module=module,
        num_workers=0,
        inference_batch_size=2,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="test")
    return datamodule


# @pytest.mark.parametrize("devices", (1, 2))
def test_index_callback(
    tmp_path: Path,
    bi_encoder_module: BiEncoderModule,
    doc_datamodule: LightningIRDataModule,
    index_config: IndexConfig,
    # devices: int,
):
    index_dir = tmp_path / "index"
    index_callback = IndexCallback(index_dir, index_config)

    trainer = LightningIRTrainer(
        # devices=devices,
        logger=False,
        enable_checkpointing=False,
        callbacks=[index_callback],
    )
    trainer.test(bi_encoder_module, datamodule=doc_datamodule)

    assert doc_datamodule.inference_datasets is not None
    assert index_callback.indexer.num_embeddings and index_callback.indexer.num_docs
    assert index_callback.indexer.num_embeddings >= index_callback.indexer.num_docs

    assert (index_dir / "index.faiss").exists() or (index_dir / "index.pt").exists()
    assert (index_dir / "doc_ids.txt").exists()
    doc_ids_path = index_dir / "doc_ids.txt"
    doc_ids = doc_ids_path.read_text().split()
    for idx, doc_id in enumerate(doc_ids):
        assert doc_id == f"doc_id_{idx+1}"
    assert (index_dir / "config.json").exists()


def get_index(
    bi_encoder_module: BiEncoderModule,
    doc_datamodule: LightningIRDataModule,
    search_config: SearchConfig,
) -> Path:
    if isinstance(search_config, FaissSearchConfig):
        index_type = "faiss"
        index_config = FaissFlatIndexConfig()
    elif isinstance(search_config, SparseSearchConfig):
        index_type = "sparse"
        index_config = SparseIndexConfig()
    else:
        raise ValueError("Unknown search_config type")
    index_dir = DATA_DIR / "indexes" / f"{index_type}-{bi_encoder_module.config.similarity_function}"
    if index_dir.exists():
        return index_dir / "lightning-ir"

    index_callback = IndexCallback(index_dir, index_config)

    trainer = LightningIRTrainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[index_callback],
    )
    trainer.test(bi_encoder_module, datamodule=doc_datamodule)
    return index_dir / "lightning-ir"


@pytest.mark.parametrize(
    "search_config",
    (
        FaissSearchConfig(k=5, imputation_strategy="min", candidate_k=10),
        FaissSearchConfig(k=5, imputation_strategy="gather", candidate_k=10),
        SparseSearchConfig(k=5),
    ),
    ids=["FaissMin", "FaissGather", "Sparse"],
)
def test_search_callback(
    tmp_path: Path,
    bi_encoder_module: BiEncoderModule,
    query_datamodule: LightningIRDataModule,
    doc_datamodule: LightningIRDataModule,
    search_config: SearchConfig,
):
    index_dir = get_index(bi_encoder_module, doc_datamodule, search_config)
    save_dir = tmp_path / "runs"
    search_callback = SearchCallback(index_dir, search_config, save_dir)

    trainer = LightningIRTrainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[search_callback],
        inference_mode=False,
    )
    trainer.test(bi_encoder_module, datamodule=query_datamodule)

    for dataloader in trainer.test_dataloaders:
        dataset = dataloader.dataset
        dataset_id = dataset.dataset_id.replace("/", "-")
        assert (save_dir / f"{dataset_id}.run").exists()
        run_df = pd.read_csv(
            save_dir / f"{dataset_id}.run",
            sep="\t",
            header=None,
            names=["query_id", "Q0", "doc_id", "rank", "score", "system"],
        )
        assert run_df["query_id"].nunique() == len(dataset)


def test_rerank_callback(tmp_path: Path, module: LightningIRModule, inference_datasets: Sequence[RunDataset]):
    datamodule = run_datamodule(module, inference_datasets)
    save_dir = tmp_path / "runs"
    rerank_callback = ReRankCallback(save_dir)
    trainer = LightningIRTrainer(logger=False, enable_checkpointing=False, callbacks=[rerank_callback])
    trainer.re_rank(module, datamodule)

    for dataloader in trainer.test_dataloaders:
        dataset = dataloader.dataset
        dataset_id = dataset.dataset_id.replace("/", "-")
        assert (save_dir / f"{dataset_id}.run").exists()
        run_df = pd.read_csv(
            save_dir / f"{dataset_id}.run",
            sep="\t",
            header=None,
            names=["query_id", "Q0", "doc_id", "rank", "score", "system"],
        )
        assert run_df["query_id"].nunique() == len(dataset)
