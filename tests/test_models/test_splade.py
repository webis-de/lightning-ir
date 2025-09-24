import pytest
import torch
from omegaconf import DictConfig
from splade.models.models_utils import get_model

from lightning_ir import BiEncoderModule


@pytest.mark.parametrize("hf_model", ["naver/splade-v3"], indirect=True)
def test_same_as_splade(hf_model: str):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    orig_model = get_model(
        DictConfig({"matching_type": "splade"}),
        DictConfig({"model_type_or_dir": str(hf_model)}),
    )
    orig_model.transformer_rep.eval()
    orig_query_encoding = orig_model.transformer_rep.tokenizer(query, return_tensors="pt")
    orig_doc_encodings = orig_model.transformer_rep.tokenizer(documents, return_tensors="pt", padding=True)
    orig_query_embeddings = orig_model(q_kwargs=orig_query_encoding)["q_rep"]
    orig_doc_embeddings = orig_model(d_kwargs=orig_doc_encodings)["d_rep"]

    module = BiEncoderModule(hf_model).eval()
    with torch.inference_mode():
        output = module.score(query, documents)
    query_embeddings = output.query_embeddings
    doc_embeddings = output.doc_embeddings

    assert torch.allclose(query_embeddings.embeddings.squeeze(1), orig_query_embeddings, atol=1e-5)
    assert torch.allclose(doc_embeddings.embeddings.squeeze(1), orig_doc_embeddings, atol=1e-5)
