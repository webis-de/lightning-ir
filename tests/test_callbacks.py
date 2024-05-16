from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

import faiss
import pandas as pd
import pytest
import torch
from lightning import Trainer
from transformers import BertModel

from lightning_ir.bi_encoder.model import BiEncoderConfig, BiEncoderModel
from lightning_ir.bi_encoder.module import BiEncoderModule
from lightning_ir.cross_encoder.model import CrossEncoderConfig, CrossEncoderModel
from lightning_ir.cross_encoder.module import CrossEncoderModule
from lightning_ir.data.datamodule import LightningIRDataModule
from lightning_ir.data.dataset import RunDataset
from lightning_ir.lightning_utils.callbacks import (
    IndexCallback,
    ReRankCallback,
    SearchCallback,
)

DATA_DIR = Path(__file__).parent / "data"


class TestBiEncoderModel(BiEncoderModel):
    def __init__(self, model_name_or_path: Path | str) -> None:
        config = BiEncoderConfig.from_pretrained(model_name_or_path)
        config.num_hidden_layers = 1
        super().__init__(config, "bert")
        self.bert = BertModel.from_pretrained(
            model_name_or_path, config=config, add_pooling_layer=False
        )
        vocab_size = self.config.vocab_size
        if self.config.add_marker_tokens:
            vocab_size += 2
        self.encoder.resize_token_embeddings(vocab_size, 8)
        self.linear = torch.nn.Linear(
            self.config.hidden_size,
            self.config.embedding_dim,
            bias=self.config.linear_bias,
        )


class TestCrossEncoderModel(CrossEncoderModel):
    config_class = CrossEncoderConfig

    def __init__(self, model_name_or_path: Path | str) -> None:
        config = CrossEncoderConfig.from_pretrained(model_name_or_path)
        config.num_hidden_layers = 1
        super().__init__(config, "bert")
        self.bert = BertModel.from_pretrained(
            config.name_or_path, config=config, add_pooling_layer=False
        )


@pytest.fixture(scope="module")
def bi_encoder_model(model_name_or_path: str) -> BiEncoderModel:
    return TestBiEncoderModel(model_name_or_path)


@pytest.fixture(scope="module")
def bi_encoder_module(bi_encoder_model: BiEncoderModel) -> BiEncoderModule:
    return BiEncoderModule(bi_encoder_model)


@pytest.fixture(scope="module")
def cross_encoder_model(model_name_or_path: str) -> CrossEncoderModel:
    return TestCrossEncoderModel(model_name_or_path)


@pytest.fixture(scope="module")
def cross_encoder_module(cross_encoder_model: CrossEncoderModel) -> CrossEncoderModule:
    return CrossEncoderModule(cross_encoder_model)


def run_datamodule(
    model: BiEncoderModel | CrossEncoderModel, inference_datasets: Sequence[RunDataset]
) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(
        model_name_or_path=model.config.name_or_path,
        config=model.config,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
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
    index_path = index_dir / "msmarco-passage"
    index_callback = IndexCallback(index_dir, 1024, num_centroids=16)

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
        assert int(doc_id) == idx
    assert (index_path / "doc_lengths.pt").exists()
    assert (index_path / "config.json").exists()
    if similarity == "l2":
        assert index_callback.indexer.index.metric_type == faiss.METRIC_L2
    elif similarity in ("cosine", "dot"):
        assert index_callback.indexer.index.metric_type == faiss.METRIC_INNER_PRODUCT


@pytest.mark.parametrize("similarity", ("cosine", "dot"))
@pytest.mark.parametrize("imputation_strategy", ("min", "gather"))
def test_search_callback(
    tmp_path: Path,
    bi_encoder_module: BiEncoderModule,
    query_datamodule: LightningIRDataModule,
    similarity: Literal["cosine", "dot", "l2"],
    imputation_strategy: Literal["min", "gather"],
):
    bi_encoder_module.config.similarity_function = similarity
    save_dir = tmp_path / "runs"
    index_path = Path(__file__).parent / "data" / f"{similarity}-index"

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


@pytest.mark.parametrize("module_name", ("bi_encoder_module", "cross_encoder_module"))
def test_rerank_callback(
    tmp_path: Path,
    module_name: str,
    inference_datasets: Sequence[RunDataset],
    request: pytest.FixtureRequest,
):
    module = request.getfixturevalue(module_name)
    datamodule = run_datamodule(module.model, inference_datasets)
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
