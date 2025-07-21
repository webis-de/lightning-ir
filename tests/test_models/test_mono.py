import torch
import transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lightning_ir import CrossEncoderModule


def test_same_as_mono():
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    model_name = "castorini/monobert-large-msmarco-finetune-only"

    orig_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    orig_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    orig_model.eval()

    enc = orig_tokenizer(
        [query] * len(documents),
        documents,
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    enc = {k: v for k, v in enc.items()}

    with torch.no_grad():
        logits = orig_model(**enc).logits

    probs = torch.nn.functional.log_softmax(logits, dim=-1)
    scores = probs[:, -1]

    module = CrossEncoderModule(model_name_or_path=model_name).eval()

    with torch.inference_mode():
        output = module.score(queries=query, docs=documents)

    assert torch.allclose(output.scores, scores, atol=1e-4)
