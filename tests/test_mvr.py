import torch
import pytest

from lightning_ir.models.mvr.config import MVRConfig
from lightning_ir.models.mvr.module import MVRModule


def test_mvr():
    model_name = "bert-base-uncased"
    config = MVRConfig()
    model = MVRModule(model_name, config=config).eval()

    query = "What is the capital of France"
    docs = ["The Capital of France is Paris", "Marseille is the capital of France"]

    with torch.inference_mode():
        output = model.score(query, docs)

    print("Similarity scores:")
    print(output.scores)
    # print("Query embeddings shape:", outputquery_embeddings.embeddings.shape)
    # print("Docs embeddings shape:", output.doc_embeddings.embeddings.shape)
