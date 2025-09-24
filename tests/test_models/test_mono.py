import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

from lightning_ir import CrossEncoderModule


@pytest.mark.parametrize("hf_model", ["castorini/monobert-large-msmarco-finetune-only"], ids=["monobert"])
def test_same_as_mono(hf_model: str):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    orig_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    orig_model = AutoModelForSequenceClassification.from_pretrained(hf_model).eval()
    module = CrossEncoderModule(model_name_or_path=hf_model).eval()

    enc = orig_tokenizer(
        [query] * len(documents), documents, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

    with torch.inference_mode():
        logits = orig_model(**enc).logits
        orig_scores = torch.nn.functional.log_softmax(logits, dim=-1)[:, 1]
        output = module.score(queries=query, docs=documents)

    assert torch.allclose(output.scores, orig_scores, atol=1e-4)


@pytest.mark.parametrize(
    "hf_model", ["castorini/monot5-base-msmarco-10k", "Soyoung97/RankT5-base"], ids=["monot5", "rankt5"]
)
def test_same_as_t5(hf_model: str):
    orig_model = T5ForConditionalGeneration.from_pretrained(hf_model).eval()
    orig_tokenizer = AutoTokenizer.from_pretrained(hf_model)

    module = CrossEncoderModule(model_name_or_path=hf_model).eval()

    query = "What is the capital of France?"
    docs = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    mode = "monot5" if "monot5" in module.config.model_type else "rankt5"

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
        scores = torch.nn.functional.log_softmax(orig_output.scores[0][:, [6136, 1176]], dim=-1)[:, 1]
    elif mode == "rankt5":
        scores = orig_output.scores[0][:, 32089]
    assert torch.allclose(output.scores, scores, atol=1e-4)
