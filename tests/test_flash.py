import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from mvr.flash.flash_model import FlashClassFactory


@pytest.mark.parametrize(
    "model_name",
    [
        "bert-base-uncased",
        "google/electra-base-discriminator",
    ],
)
def test_same_as_model(model_name: str) -> None:
    config = AutoConfig.from_pretrained(model_name)
    model_class = AutoModel._model_mapping[type(config)]
    FlashModel = FlashClassFactory(model_class)
    flash_model = FlashModel.from_pretrained(model_name)
    flash_model = flash_model.eval()
    base_model = model_class.from_pretrained(model_name)
    base_model = base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    queries = [
        "What is the meaning of life?",
        "What is the meaning of life?",
        "What is the meaning of death?",
    ]
    docs = [
        "The meaning of life is to be happy.",
        "The meaning of life is to be happy.",
        "Death is meaningless.",
    ]
    encoded = tokenizer(queries, docs, return_tensors="pt", padding=True)

    with torch.no_grad():
        base_output = base_model(**encoded)
        flash_model_output = flash_model(**encoded)

    assert torch.allclose(
        base_output.last_hidden_state,
        flash_model_output.last_hidden_state,
        atol=1e-4,
        rtol=1e-4,
    )
