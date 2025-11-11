import pytest
import torch
from sentence_transformers import SentenceTransformer, util

from lightning_ir import BiEncoderModule


@pytest.mark.model
@pytest.mark.parametrize(
    "hf_model",
    [
        "sentence-transformers/msmarco-bert-base-dot-v5",
        "sentence-transformers/msmarco-MiniLM-L-6-v3",
    ],
    indirect=True,
)
def test_same_as_sentence_transformer(hf_model: str):
    query = "This is an example query"
    docs = ["This is an example sentence", "Each sentence is converted"]

    orig_model = SentenceTransformer(hf_model)
    orig_query_embeddings = orig_model.encode(query)
    orig_doc_embeddings = orig_model.encode(docs)
    orig_scores = util.dot_score(torch.from_numpy(orig_query_embeddings), torch.from_numpy(orig_doc_embeddings))

    module = BiEncoderModule(hf_model).eval()
    with torch.no_grad():
        output = module.score(query, docs)

    assert torch.allclose(
        output.query_embeddings.embeddings.squeeze(1),
        torch.from_numpy(orig_query_embeddings),
        atol=1e-5,
    )
    assert torch.allclose(
        output.doc_embeddings.embeddings.squeeze(1),
        torch.from_numpy(orig_doc_embeddings),
        atol=1e-5,
    )
    assert torch.allclose(output.scores, orig_scores.squeeze(0), atol=1e-5)
