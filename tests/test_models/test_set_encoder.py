import torch

from lightning_ir import CrossEncoderModule


def test_set_encoder():
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    model_name = "webis/set-encoder-base"
    module = CrossEncoderModule(model_name).eval()
    with torch.inference_mode():
        output = module.score(query, documents)

    assert output.scores[0] > output.scores[1]
