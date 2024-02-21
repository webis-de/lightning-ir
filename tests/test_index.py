from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from lightning import Trainer
from transformers import BertModel

from mvr.callbacks import IndexCallback, IndexConfig, SearchCallback, SearchConfig
from mvr.datamodule import MVRDataModule
from mvr.mvr import MVRConfig, MVRModel, MVRModule


class TestModel(MVRModel):
    def __init__(self, model_name_or_path: Path | str) -> None:
        config = MVRConfig.from_pretrained(model_name_or_path)
        super().__init__(config)
        self.bert = BertModel.from_pretrained(
            model_name_or_path, config=self.config, add_pooling_layer=False
        )
        vocab_size = self.config.vocab_size
        if self.config.add_marker_tokens:
            vocab_size += 2
        self.bert.resize_token_embeddings(vocab_size, 8)
        self.linear = torch.nn.Linear(
            self.config.hidden_size,
            self.config.embedding_dim,
            bias=self.config.linear_bias,
        )

    @property
    def encoder(self) -> torch.nn.Module:
        return self.bert


@pytest.fixture(scope="module")
def mvr_model(model_name_or_path: str) -> MVRModel:
    return TestModel(model_name_or_path)


@pytest.fixture(scope="module")
def mvr_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model)


# @pytest.mark.parametrize("devices", (1, 2))
@pytest.mark.parametrize("devices", (1,))
@pytest.mark.parametrize("save_doc_lengths", (True, False))
def test_indexing(
    tmp_path: Path,
    mvr_module: MVRModule,
    doc_datamodule: MVRDataModule,
    devices: int,
    save_doc_lengths: bool,
):
    index_dir = tmp_path / "index"
    index_callback = IndexCallback(
        IndexConfig(
            index_dir,
            1024,
            save_doc_lengths=save_doc_lengths,
            num_centroids=16,
        )
    )

    trainer = Trainer(
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        callbacks=[index_callback],
    )
    trainer.predict(mvr_module, datamodule=doc_datamodule)

    assert doc_datamodule.inference_datasets is not None
    index_dir = index_dir / doc_datamodule.inference_datasets[0]
    assert index_callback._num_embeddings == 1872
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "doc_ids.npy").exists()
    doc_ids_path = index_dir / "doc_ids.npy"
    file_size = doc_ids_path.stat().st_size
    num_elements = file_size // (20 * np.dtype("uint8").itemsize)
    doc_ids = np.memmap(doc_ids_path, dtype="S20", mode="r", shape=(num_elements,))
    for idx, doc_id in enumerate(doc_ids):
        assert int(doc_id.decode("utf-8")) == idx
    if index_callback.config.save_doc_lengths:
        assert (index_dir / "doc_lengths.npy").exists()


@pytest.mark.parametrize("doc_lengths", (True, False))
@pytest.mark.parametrize("imputation_strategy", ("min", "gather"))
def test_searching(
    mvr_module: MVRModule,
    query_datamodule: MVRDataModule,
    doc_lengths: bool,
    imputation_strategy: Literal["min", "gather"],
):
    if doc_lengths:
        sub_dir = "index-with-doc_lengths"
    else:
        sub_dir = "index-without-doc_lengths"
    index_path = Path(__file__).parent / "data" / sub_dir / "msmarco-passage"

    if imputation_strategy == "gather" and not doc_lengths:
        with pytest.raises(ValueError):
            search_callback = SearchCallback(
                SearchConfig(index_path, 10, imputation_strategy)
            )
        return

    search_callback = SearchCallback(SearchConfig(index_path, 10, imputation_strategy))

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[search_callback],
    )
    trainer.predict(mvr_module, datamodule=query_datamodule)
