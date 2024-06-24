from pathlib import Path
from typing import Literal, Sequence

import faiss
import pandas as pd
import pytest
from lightning import Trainer

from lightning_ir import (
    BiEncoderModule,
    FlatIndexConfig,
    LightningIRDataModule,
    LightningIRModule,
    RunDataset,
)
from lightning_ir.lightning_utils.callbacks import (
    IndexCallback,
    ReRankCallback,
    SearchCallback,
)


def run_datamodule(
    module: LightningIRModule, inference_datasets: Sequence[RunDataset]
) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        module=module,
        num_workers=0,
        inference_batch_size=2,
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="predict")
    return datamodule


# @pytest.mark.parametrize("devices", (1, 2))
@pytest.mark.parametrize("similarity", ("cosine", "dot", "l2"))
@pytest.mark.parametrize("devices", (1,))
def test_index_callback(
    tmp_path: Path,
    bi_encoder_module: BiEncoderModule,
    doc_datamodule: LightningIRDataModule,
    similarity: Literal["cosine", "dot", "l2"],
    devices: int,
):
    bi_encoder_module.config.similarity_function = similarity
    index_dir = tmp_path / "index"
    index_path = index_dir / doc_datamodule.inference_datasets[0].docs_dataset_id
    index_config = FlatIndexConfig()
    index_callback = IndexCallback(index_dir, index_config)

    trainer = Trainer(
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        callbacks=[index_callback],
    )
    trainer.predict(bi_encoder_module, datamodule=doc_datamodule)

    assert doc_datamodule.inference_datasets is not None
    assert index_callback.indexer.num_embeddings == index_callback.indexer.index.ntotal
    assert (index_path / "index.faiss").exists()
    assert (index_path / "doc_ids.txt").exists()
    doc_ids_path = index_path / "doc_ids.txt"
    doc_ids = doc_ids_path.read_text().split()
    for idx, doc_id in enumerate(doc_ids):
        assert doc_id == f"doc_id_{idx+1}"
    assert (index_path / "doc_lengths.pt").exists()
    assert (index_path / "config.json").exists()
    if similarity == "l2":
        assert index_callback.indexer.index.metric_type == faiss.METRIC_L2
    elif similarity in ("cosine", "dot"):
        assert index_callback.indexer.index.metric_type == faiss.METRIC_INNER_PRODUCT


@pytest.mark.parametrize("similarity", ("cosine", "dot"))
@pytest.mark.parametrize("imputation_strategy", ("min", "gather", None))
def test_search_callback(
    tmp_path: Path,
    bi_encoder_module: BiEncoderModule,
    query_datamodule: LightningIRDataModule,
    similarity: Literal["cosine", "dot", "l2"],
    imputation_strategy: Literal["min", "gather"],
):
    bi_encoder_module.config.similarity_function = similarity
    save_dir = tmp_path / "runs"
    index_path = Path(__file__).parent / "data" / "indexes" / f"{similarity}-index"

    search_callback = SearchCallback(save_dir, index_path, 5, 10, imputation_strategy)

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[search_callback],
    )
    trainer.predict(bi_encoder_module, datamodule=query_datamodule)

    for dataloader in trainer.predict_dataloaders:
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


def test_rerank_callback(
    tmp_path: Path,
    module: LightningIRModule,
    inference_datasets: Sequence[RunDataset],
):
    datamodule = run_datamodule(module, inference_datasets)
    save_dir = tmp_path / "runs"
    rerank_callback = ReRankCallback(save_dir)
    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[rerank_callback],
    )
    trainer.predict(module, datamodule=datamodule)

    for dataloader in trainer.predict_dataloaders:
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
