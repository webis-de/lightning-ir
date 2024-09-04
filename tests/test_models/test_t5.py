import pytest
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from lightning_ir import CrossEncoderModule, T5CrossEncoderConfig


@pytest.mark.parametrize(
    "model_name", ["castorini/monot5-base-msmarco-10k", "Soyoung97/RankT5-base"], ids=["monot5", "rankt5"]
)
def test_same_as_t5(model_name: str):
    orig_model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    orig_tokenizer = AutoTokenizer.from_pretrained(model_name)

    module = CrossEncoderModule(model_name_or_path=model_name).eval()

    query = "What is the capital of France?"
    docs = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    mode = "monot5" if "monot5" in model_name else "rankt5"

    if mode == "monot5":
        input_texts = [f"Query: {query} Document: {doc} Relevant:" for doc in docs]
    elif mode == "rankt5":
        input_texts = [f"Query: {query} Document: {doc}" for doc in docs]
    else:
        raise ValueError("unknown model type")

    orig_encoded = orig_tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.inference_mode():
        orig_output = orig_model.generate(
            **orig_encoded, max_length=2, return_dict_in_generate=True, output_scores=True
        )
        output = module.score(queries=query, docs=docs)
    if mode == "monot5":
        scores = torch.nn.functional.log_softmax(orig_output.scores[0][:, [1176, 6136]], dim=1)[:, 0]
    elif mode == "rankt5":
        scores = orig_output.scores[0][:, 32089]
    assert torch.allclose(output.scores, scores, atol=1e-4)
