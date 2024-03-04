from pathlib import Path
from typing import Literal
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import pytest
import torch
from lightning import Trainer
from transformers import BertModel

from mvr.callbacks import IndexCallback, SearchCallback
from mvr.datamodule import MVRDataModule
from mvr.mvr import MVRConfig, MVRModel, MVRModule


class TestModel(MVRModel):
    def __init__(self, model_name_or_path: Path | str) -> None:
        config = MVRConfig.from_pretrained(model_name_or_path)
        bert = BertModel.from_pretrained(
            model_name_or_path, config=config, add_pooling_layer=False
        )
        super().__init__(config, bert)
        vocab_size = self.config.vocab_size
        if self.config.add_marker_tokens:
            vocab_size += 2
        self.encoder.resize_token_embeddings(vocab_size, 8)
        self.linear = torch.nn.Linear(
            self.config.hidden_size,
            self.config.embedding_dim,
            bias=self.config.linear_bias,
        )


@pytest.fixture(scope="module")
def mvr_model(model_name_or_path: str) -> MVRModel:
    return TestModel(model_name_or_path)


@pytest.fixture(scope="module")
def mvr_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model)


# @pytest.mark.parametrize("devices", (1, 2))
@pytest.mark.parametrize("similarity", ("cosine", "dot", "l2"))
@pytest.mark.parametrize("devices", (1,))
def test_index_callback(
    tmp_path: Path,
    mvr_module: MVRModule,
    doc_datamodule: MVRDataModule,
    similarity: Literal["cosine", "dot", "l2"],
    devices: int,
):
    mvr_module.config.similarity_function = similarity
    index_path = tmp_path / "index"
    index_callback = IndexCallback(index_path, 1024, num_centroids=16)

    trainer = Trainer(
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        callbacks=[index_callback],
    )
    trainer.predict(mvr_module, datamodule=doc_datamodule)

    assert doc_datamodule.inference_datasets is not None
    assert index_callback.indexer.num_embeddings == index_callback.indexer.index.ntotal
    assert (index_path / "index.faiss").exists()
    assert (index_path / "doc_ids.pt").exists()
    doc_ids_path = index_path / "doc_ids.pt"
    doc_ids = torch.load(doc_ids_path)
    for idx, doc_id in enumerate(doc_ids):
        assert int(bytes(doc_id).decode("utf-8")) == idx
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
    mvr_module: MVRModule,
    query_datamodule: MVRDataModule,
    similarity: Literal["cosine", "dot", "l2"],
    imputation_strategy: Literal["min", "gather"],
):
    mvr_module.config.similarity_function = similarity
    save_dir = tmp_path / "runs"
    index_path = Path(__file__).parent / "data" / f"{similarity}-index"

    search_callback = SearchCallback(save_dir, index_path, 5, 10, imputation_strategy)

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[search_callback],
    )
    trainer.predict(mvr_module, datamodule=query_datamodule)

    for dataloader in trainer.predict_dataloaders:
        dataset = dataloader.dataset
        dataset_id = dataset.ir_dataset.dataset_id().replace("/", "-")
        assert (save_dir / f"{dataset_id}.run").exists()
        run_df = pd.read_csv(
            save_dir / f"{dataset_id}.run",
            sep="\t",
            header=None,
            names=["query_id", "Q0", "doc_id", "rank", "score", "system"],
        )
        assert run_df["query_id"].nunique() == len(dataset)
