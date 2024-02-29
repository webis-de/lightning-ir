from pathlib import Path
from typing import Literal

import pytest
import torch
from transformers import BertModel

from mvr.datamodule import MVRDataModule
from mvr.loss import LocalizedContrastive, MarginMSE, RankNet
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


@pytest.fixture()
def xtr_model(model_name_or_path: str) -> MVRModel:
    model = TestModel(model_name_or_path)
    model.config.xtr_token_retrieval_k = 32
    return model


@pytest.fixture(scope="module")
def margin_mse_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model, MarginMSE())


@pytest.fixture(scope="module")
def ranknet_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model, RankNet())


@pytest.fixture(scope="module")
def localized_contrastive_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model, LocalizedContrastive())


@pytest.fixture()
def in_batch_negatives_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model, MarginMSE(in_batch_loss="ce"))


@pytest.fixture()
def xtr_module(xtr_model: MVRModel) -> MVRModule:
    return MVRModule(xtr_model, MarginMSE())


def test_doc_padding(relevance_run_datamodule: MVRDataModule, mvr_model: MVRModel):
    batch = next(iter(relevance_run_datamodule.train_dataloader()))
    model = mvr_model
    doc_encoding = batch.doc_encoding
    doc_encoding["input_ids"] = doc_encoding["input_ids"][:-1]
    doc_encoding["attention_mask"] = doc_encoding["attention_mask"][:-1]
    doc_encoding["token_type_ids"] = doc_encoding["token_type_ids"][:-1]

    query_embedding = model.encode_queries(**batch.query_encoding)
    doc_embedding = model.encode_docs(**batch.doc_encoding)
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            doc_embedding,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
            None,
        )
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            doc_embedding,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
            [doc_embedding.shape[0]],
        )
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            doc_embedding,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
            [0] * query_embedding.shape[0],
        )

    num_docs = [len(docs) for docs in batch.doc_ids]
    num_docs[-1] = num_docs[-1] - 1
    query_scoring_mask, doc_scoring_mask = model.scoring_masks(
        batch.query_encoding.input_ids,
        batch.doc_encoding.input_ids,
        batch.query_encoding.attention_mask,
        batch.doc_encoding.attention_mask,
    )
    scores = model.score(
        query_embedding,
        doc_embedding,
        query_scoring_mask,
        doc_scoring_mask,
        num_docs,
    )
    assert scores.shape[0] == doc_embedding.shape[0]


@pytest.mark.parametrize("similarity_function", ["cosine", "dot", "l2"])
def test_margin_mse(
    similarity_function: Literal["cosine", "dot", "l2"],
    margin_mse_module: MVRModule,
    tuples_datamodule: MVRDataModule,
):
    dataloader = tuples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    margin_mse_module.config.similarity_function = similarity_function
    loss = margin_mse_module.training_step(batch, 0)
    assert loss


@pytest.mark.parametrize("similarity_function", ["cosine", "dot", "l2"])
def test_ranknet(
    similarity_function: Literal["cosine", "dot", "l2"],
    ranknet_module: MVRModule,
    rank_run_datamodule: MVRDataModule,
):
    dataloader = rank_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    ranknet_module.config.similarity_function = similarity_function
    loss = ranknet_module.training_step(batch, 0)
    assert loss


@pytest.mark.parametrize("similarity_function", ["cosine", "dot", "l2"])
def test_localized_contrastive(
    similarity_function: Literal["cosine", "dot", "l2"],
    localized_contrastive_module: MVRModule,
    single_relevant_run_datamodule: MVRDataModule,
):
    dataloader = single_relevant_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    localized_contrastive_module.config.similarity_function = similarity_function
    loss = localized_contrastive_module.training_step(batch, 0)
    assert loss


def test_in_batch_negatives(
    in_batch_negatives_module: MVRModule, tuples_datamodule: MVRDataModule
):
    dataloader = tuples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = in_batch_negatives_module.training_step(batch, 0)
    assert loss


def test_xtr(xtr_module: MVRModule, tuples_datamodule: MVRDataModule):
    dataloader = tuples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = xtr_module.training_step(batch, 0)
    assert loss


def test_validation_step(
    margin_mse_module: MVRModule,
    relevance_run_datamodule: MVRDataModule,
):
    dataloader = relevance_run_datamodule.val_dataloader()[0]
    batch = next(iter(dataloader))
    margin_mse_module.validation_step(batch, 0, 0)
    outputs = margin_mse_module.validation_step_outputs
    assert len(outputs) == 2
    assert outputs[0][0] == "ndcg@10"
    assert outputs[1][0] == "mrr@ranking"
