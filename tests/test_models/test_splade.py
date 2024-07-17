import torch
from omegaconf import DictConfig
from splade.models.models_utils import get_model

from lightning_ir import SpladeModel


def test_same_as_splade():
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    model_name = "naver/splade-v3"
    orig_model = get_model(
        DictConfig({"matching_type": "splade"}),
        DictConfig({"model_type_or_dir": model_name}),
    )
    orig_model.transformer_rep.eval()
    orig_query_encoding = orig_model.transformer_rep.tokenizer(query, return_tensors="pt")
    orig_doc_encodings = orig_model.transformer_rep.tokenizer(documents, return_tensors="pt", padding=True)
    orig_query_embeddings = orig_model(q_kwargs=orig_query_encoding)["q_rep"]
    orig_doc_embeddings = orig_model(d_kwargs=orig_doc_encodings)["d_rep"]

    model = SpladeModel.from_pretrained(model_name).eval()
    tokenizer = SpladeModel.config_class.tokenizer_class.from_pretrained(model_name, **model.config.to_tokenizer_dict())
    query_encoding = tokenizer.tokenize_query(query, return_tensors="pt")
    doc_encoding = tokenizer.tokenize_doc(documents, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embeddings = model.encode_doc(**query_encoding)
        doc_embeddings = model.encode_query(**doc_encoding)

    assert torch.allclose(query_embeddings.embeddings.squeeze(1), orig_query_embeddings, atol=1e-6)
    assert torch.allclose(doc_embeddings.embeddings.squeeze(1), orig_doc_embeddings, atol=1e-5)
