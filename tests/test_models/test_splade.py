import pytest
import torch
from sentence_transformers import SparseEncoder

from lightning_ir import BiEncoderModule


@pytest.mark.parametrize(
    "hf_model",
    ["naver/splade-v3", "naver/splade-v3-distilbert", "naver/splade-v3-doc", "naver/splade-v3-lexical"],
    indirect=True,
)
def test_same_as_splade(hf_model: str):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    orig_model = SparseEncoder(hf_model)
    orig_query_embeddings = orig_model.encode_query([query])
    orig_doc_embeddings = orig_model.encode_document(documents)
    orig_scores = orig_model.similarity(orig_query_embeddings, orig_doc_embeddings)

    module = BiEncoderModule(hf_model).eval()
    with torch.inference_mode():
        output = module.score(query, documents)
    query_embeddings = output.query_embeddings
    doc_embeddings = output.doc_embeddings
    scores = output.scores

    assert torch.allclose(query_embeddings.embeddings.squeeze(1), orig_query_embeddings.to_dense().cpu(), atol=1e-4)
    assert torch.allclose(doc_embeddings.embeddings.squeeze(1), orig_doc_embeddings.to_dense().cpu(), atol=1e-4)
    assert torch.allclose(scores, orig_scores.cpu().view(-1), atol=1e-5)
