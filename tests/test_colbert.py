import pytest
import torch
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import colbert_score

from mvr.colbert import ColBERTConfig, ColBERTModel, ColBERTModule
from mvr.datamodule import MVRDataModule
from mvr.loss import SupervisedMarginMSE
from mvr.tokenizer import MVRTokenizer


@pytest.fixture(scope="module")
def colbert_model(model_name_or_path: str) -> ColBERTModel:
    config = ColBERTConfig.from_pretrained(model_name_or_path)
    model = ColBERTModel.from_pretrained(model_name_or_path, config=config)
    return model


@pytest.fixture(scope="module")
def colbert_module(model_name_or_path: str) -> ColBERTModule:
    config = ColBERTConfig.from_pretrained(model_name_or_path)
    return ColBERTModule(model_name_or_path, config, [SupervisedMarginMSE()])


def test_training_step(colbert_module: ColBERTModule, tuples_datamodule: MVRDataModule):
    dataloader = tuples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = colbert_module.training_step(batch, 0)
    assert loss


def test_seralize_deserialize(
    colbert_model: ColBERTModel, tmpdir_factory: pytest.TempdirFactory
):
    save_dir = tmpdir_factory.mktemp("colbert")
    colbert_model.save_pretrained(save_dir)
    new_model = ColBERTModel.from_pretrained(save_dir, mask_punctuation=False)
    for key, value in colbert_model.config.__dict__.items():
        if key in (
            "torch_dtype",
            "_name_or_path",
            "_commit_hash",
            "transformers_version",
            "model_type",
        ):
            continue
        if key == "mask_punctuation":
            assert value and not getattr(new_model.config, key)
            continue
        assert getattr(new_model.config, key) == value
    for key, value in colbert_model.state_dict().items():
        assert new_model.state_dict()[key].equal(value)


def test_same_as_colbert():
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    model = ColBERTModel.from_colbert_checkpoint("colbert-ir/colbertv2.0")
    tokenizer = MVRTokenizer.from_pretrained(
        "colbert-ir/colbertv2.0", **model.config.to_dict()
    )
    query_encoding = tokenizer.tokenize_queries(query, return_tensors="pt")
    doc_encoding = tokenizer.tokenize_docs(
        documents, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        query_embedding = model.encode_queries(
            query_encoding.input_ids, query_encoding.attention_mask
        )
        doc_embedding = model.encode_docs(
            doc_encoding.input_ids, doc_encoding.attention_mask
        )
    query_scoring_mask, doc_scoring_mask = model.scoring_masks(
        query_encoding.input_ids,
        doc_encoding.input_ids,
        query_encoding.attention_mask,
        doc_encoding.attention_mask,
    )
    scores = model.score(
        query_embedding, doc_embedding, query_scoring_mask, doc_scoring_mask, None
    )

    orig_model = Checkpoint("colbert-ir/colbertv2.0")
    orig_query = orig_model.queryFromText([query])
    orig_docs = orig_model.docFromText(documents)
    d_mask = ~(orig_docs == 0).all(-1)
    orig_scores = colbert_score(orig_query, orig_docs, d_mask)

    iterator = zip(model.state_dict().items(), orig_model.state_dict().items())
    for (key, weight), (orig_key, orig_weight) in iterator:
        assert key == orig_key[6:]
        if "word_embeddings" not in key:
            assert torch.allclose(weight, orig_weight)

    assert torch.allclose(query_embedding, orig_query)
    assert torch.allclose(doc_embedding[d_mask], orig_docs[d_mask])
    assert torch.allclose(scores, orig_scores)
