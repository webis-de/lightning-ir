import torch

from lightning_ir import BiEncoderModule
from lightning_ir.models import MvrConfig


def test_mvr():
    model_name = "bert-base-uncased"
    config = MvrConfig()
    model = BiEncoderModule(model_name, config=config).eval()

    query = "What is the capital of France"
    docs = ["The Capital of France is Paris", "Marseille is the capital of France"]

    with torch.inference_mode():
        output = model.score(query, docs)

    assert output.query_embeddings.embeddings.shape[1] == 1
    assert output.doc_embeddings.embeddings.shape[1] == config.num_viewer_tokens
    assert output.scores.shape[0] == 2

    for i in range(model.config.num_viewer_tokens):
        viewer_token = model.tokenizer.viewer_token_id(i)
        assert viewer_token is not None
        assert (output.doc_embeddings.encoding["input_ids"][:, i + 1] == viewer_token).all()
