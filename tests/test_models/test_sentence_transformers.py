import pytest
import torch
from sentence_transformers import SentenceTransformer, util

from lightning_ir import BiEncoderModel


@pytest.mark.parametrize(
    "model_name",
    [
        "sentence-transformers/msmarco-bert-base-dot-v5",
        "sentence-transformers/msmarco-MiniLM-L-6-v3",
    ],
    ids=["bert", "minilm"],
)
def test_same_as_sentence_transformer(model_name: str):
    query = "This is an example query"
    docs = ["This is an example sentence", "Each sentence is converted"]

    orig_model = SentenceTransformer(model_name)
    orig_query_embeddings = orig_model.encode(query)
    orig_doc_embeddings = orig_model.encode(docs)
    orig_scores = util.dot_score(
        torch.from_numpy(orig_query_embeddings), torch.from_numpy(orig_doc_embeddings)
    )

    model = BiEncoderModel.from_pretrained(
        model_name,
        add_marker_tokens=False,
        projection=None,
        embedding_dim=orig_query_embeddings.shape[-1],
    )
    tokenizer = BiEncoderModel.config_class.tokenizer_class.from_pretrained(
        model_name, **model.config.to_tokenizer_dict()
    )
    encodings = tokenizer.tokenize(query, docs, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model.forward(encodings["query_encoding"], encodings["doc_encoding"])

    assert torch.allclose(
        output.query_embeddings.embeddings.squeeze(1),
        torch.from_numpy(orig_query_embeddings),
        atol=1e-6,
    )
    assert torch.allclose(
        output.doc_embeddings.embeddings.squeeze(1),
        torch.from_numpy(orig_doc_embeddings),
        atol=1e-6,
    )
    assert torch.allclose(output.scores, orig_scores.squeeze(0), atol=1e-6)
