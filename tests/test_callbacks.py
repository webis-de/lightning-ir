from pathlib import Path
from typing import Sequence

import ir_datasets
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest

from lightning_ir import BiEncoderModule, LightningIRDataModule, LightningIRModule, LightningIRTrainer, RunDataset
from lightning_ir.callbacks import IndexCallback, RegisterLocalDatasetCallback, ReRankCallback, SearchCallback
from lightning_ir.models import DprConfig
from lightning_ir.retrieve import (
    FaissFlatIndexConfig,
    FaissIVFIndexConfig,
    FaissSearchConfig,
    IndexConfig,
    PlaidIndexConfig,
    PlaidSearchConfig,
    SearchConfig,
    SeismicIndexConfig,
    SeismicSearchConfig,
    TorchDenseIndexConfig,
    TorchDenseSearchConfig,
    TorchSparseIndexConfig,
    TorchSparseSearchConfig,
)

from .conftest import CORPUS_DIR, DATA_DIR


@pytest.fixture(
    params=[
        FaissFlatIndexConfig(),
        FaissIVFIndexConfig(num_centroids=16),
        TorchSparseIndexConfig(),
        TorchDenseIndexConfig(),
        PlaidIndexConfig(num_centroids=8, num_train_embeddings=1_024),
        SeismicIndexConfig(num_postings=32),
    ],
    ids=["Faiss", "FaissIVF", "Sparse", "Dense", "Plaid", "Seismic"],
)
def index_config(request: SubRequest) -> IndexConfig:
    return request.param


def run_datamodule(module: LightningIRModule, inference_datasets: Sequence[RunDataset]) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(num_workers=0, inference_batch_size=2, inference_datasets=inference_datasets)
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
    if bi_encoder_module.config.model_type not in index_config.SUPPORTED_MODELS:
        pytest.skip(
            f"Indexing not supported for {bi_encoder_module.config.__class__.__name__} model "
            f"and {index_config.__class__.__name__} indexer"
        )

    index_dir = tmp_path / "index"
    index_callback = IndexCallback(index_config=index_config, index_dir=index_dir)

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

    dataset_id = doc_datamodule.inference_datasets[0].dataset_id
    index_dir = index_dir / dataset_id
    assert (
        (index_dir / "index.faiss").exists()  # faiss
        or (index_dir / "index.pt").exists()  # sparse
        or (index_dir / "centroids.pt").exists()  # plaid
        or (index_dir / ".index.seismic").exists()  # seismic
    )
    assert (index_dir / "doc_ids.txt").exists()
    doc_ids_path = index_dir / "doc_ids.txt"
    doc_ids = doc_ids_path.read_text().split()
    for idx, doc_id in enumerate(doc_ids):
        assert doc_id == f"doc_id_{idx + 1}"
    assert (index_dir / "config.json").exists()


def get_index(
    bi_encoder_module: BiEncoderModule,
    doc_datamodule: LightningIRDataModule,
    search_config: SearchConfig,
) -> Path:
    index_config: IndexConfig
    if isinstance(search_config, FaissSearchConfig):
        index_type = "faiss"
        index_config = FaissFlatIndexConfig()
    elif isinstance(search_config, TorchSparseSearchConfig):
        index_type = "sparse"
        index_config = TorchSparseIndexConfig()
    elif isinstance(search_config, TorchDenseSearchConfig):
        index_type = "dense"
        index_config = TorchDenseIndexConfig()
    elif isinstance(search_config, PlaidSearchConfig):
        index_type = "plaid"
        index_config = PlaidIndexConfig(num_centroids=8, num_train_embeddings=1_024)
    elif isinstance(search_config, SeismicSearchConfig):
        index_type = "seismic"
        index_config = SeismicIndexConfig(num_postings=32)
    else:
        raise ValueError("Unknown search_config type")
    index_dir = (
        DATA_DIR
        / "indexes"
        / f"{index_type}-{bi_encoder_module.config.model_type}-{bi_encoder_module.config.similarity_function}"
    )
    if index_dir.exists():
        return index_dir

    index_callback = IndexCallback(index_config=index_config, index_dir=index_dir)

    trainer = LightningIRTrainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[index_callback],
    )
    trainer.test(bi_encoder_module, datamodule=doc_datamodule)
    return index_dir


@pytest.mark.parametrize(
    "search_config",
    (
        FaissSearchConfig(k=3, imputation_strategy="min", candidate_k=3),
        FaissSearchConfig(k=3, imputation_strategy="gather", candidate_k=3),
        PlaidSearchConfig(k=3, centroid_score_threshold=0),
        TorchSparseSearchConfig(k=3),
        TorchDenseSearchConfig(k=3),
        SeismicSearchConfig(k=3),
    ),
    ids=["FaissMin", "FaissGather", "Plaid", "Sparse", "Dense", "Seismic"],
)
def test_search_callback(
    tmp_path: Path,
    bi_encoder_module: BiEncoderModule,
    query_datamodule: LightningIRDataModule,
    doc_datamodule: LightningIRDataModule,
    search_config: SearchConfig,
):

    if bi_encoder_module.config.model_type not in search_config.SUPPORTED_MODELS:
        pytest.skip(
            f"Searching not supported for {bi_encoder_module.config.__class__.__name__} model and "
            f"{search_config.__class__.__name__} searcher"
        )

    index_dir = get_index(bi_encoder_module, doc_datamodule, search_config)
    save_dir = tmp_path / "runs"
    search_callback = SearchCallback(search_config=search_config, index_dir=index_dir, save_dir=save_dir, use_gpu=False)

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


def test_register_local_dataset_callback(model_name_or_path: str):
    callback = RegisterLocalDatasetCallback(
        dataset_id="test",
        docs=str(CORPUS_DIR / "docs.tsv"),
        queries=str(CORPUS_DIR / "queries.tsv"),
        qrels=str(CORPUS_DIR / "qrels.tsv"),
        docpairs=str(CORPUS_DIR / "docpairs.tsv"),
    )
    module = LightningIRModule(model_name_or_path=model_name_or_path, config=DprConfig(embedding_dim=4))
    datamodule = LightningIRDataModule(train_dataset=RunDataset("test"), train_batch_size=2)

    trainer = LightningIRTrainer(logger=False, enable_checkpointing=False, callbacks=[callback])

    trainer.test(module, datamodule)

    assert ir_datasets.registry._registered.get("test") is not None
