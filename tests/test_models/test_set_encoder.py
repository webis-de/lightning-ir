import pytest
import torch

from lightning_ir import CrossEncoderModule


@pytest.mark.model
@pytest.mark.parametrize(
    "hf_model",
    [
        "webis/set-encoder-base",
    ],
    indirect=True,
)
def test_set_encoder(hf_model: str):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    module = CrossEncoderModule(hf_model).eval()
    with torch.inference_mode():
        output = module.score(query, documents)

    assert output.scores[0] > output.scores[1]
