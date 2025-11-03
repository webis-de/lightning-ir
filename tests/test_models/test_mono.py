import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from torch.nn import functional as F
from lightning_ir import CrossEncoderModule


@pytest.mark.parametrize("hf_model", ["castorini/monobert-large-msmarco-finetune-only"], indirect=True)
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


@pytest.mark.parametrize("hf_model", ["castorini/monot5-base-msmarco-10k", "Soyoung97/RankT5-base"], indirect=True)
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

    if module.config.scoring_strategy == "mono":
        input_texts = [f"Query: {query} Document: {doc} Relevant:" for doc in docs]
    elif module.config.scoring_strategy == "rank":
        input_texts = [f"Query: {query} Document: {doc}" for doc in docs]
    else:
        raise ValueError("unknown model type")

    orig_encoded = orig_tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.inference_mode():
        orig_output = orig_model.generate(
            **orig_encoded, max_length=2, return_dict_in_generate=True, output_scores=True
        )
        output = module.score(queries=query, docs=docs)
    if module.config.scoring_strategy == "mono":
        scores = torch.nn.functional.log_softmax(orig_output.scores[0][:, [6136, 1176]], dim=-1)[:, 1]
    elif module.config.scoring_strategy == "rank":
        scores = orig_output.scores[0][:, 32089]
    else:
        raise ValueError("unknown model type")
    assert torch.allclose(output.scores, scores, atol=1e-4)


@pytest.mark.parametrize("hf_model", ["castorini/monot5-base-msmarco" ], indirect=True)
def test_same_as_t5_long_example(hf_model: str):
    orig_model = T5ForConditionalGeneration.from_pretrained(hf_model).eval()
    orig_tokenizer = AutoTokenizer.from_pretrained(hf_model)

    module = CrossEncoderModule(model_name_or_path=hf_model).eval()

    query = "which health care system provides all citizens or residents with equal access to health care services"
    docs = [
        "Search form. Health care is a basic human need. An effective health care system that spends wisely and covers everyone is critical for public health, safety and economic security. A single-payer health care system covers everyone. It has succeeded in countries throughout the developed world.",
        "Under a single-payer system, all residents of the U.S. would be covered for all medically necessary services, including doctor, hospital, preventive, long-term care, mental health, reproductive health care, dental, vision, prescription drug and medical supply costs.",
    ]

    if module.config.scoring_strategy == "mono":
        input_texts = [f"Query: {query} Document: {doc} Relevant:" for doc in docs]
    elif module.config.scoring_strategy == "rank":
        input_texts = [f"Query: {query} Document: {doc}" for doc in docs]
    else:
        raise ValueError("unknown model type")

    orig_encoded = orig_tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.inference_mode():
        orig_output = orig_model.generate(
            **orig_encoded, max_length=2, return_dict_in_generate=True, output_scores=True
        )
        output = module.score(queries=query, docs=docs)
    if module.config.scoring_strategy == "mono":
        scores = torch.nn.functional.log_softmax(orig_output.scores[0][:, [6136, 1176]], dim=-1)[:, 1]
    elif module.config.scoring_strategy == "rank":
        scores = orig_output.scores[0][:, 32089]
    else:
        raise ValueError("unknown model type")
    assert torch.allclose(output.scores, scores, atol=1e-4)

@pytest.mark.parametrize("hf_model", ["castorini/monot5-base-msmarco" ], indirect=True)
def test_same_as_pyterriert5(hf_model: str):
    orig_model = T5ForConditionalGeneration.from_pretrained(hf_model).eval()
    orig_tokenizer = T5Tokenizer.from_pretrained(hf_model)

    module = CrossEncoderModule(model_name_or_path=hf_model).eval()

    query = "which health care system provides all citizens or residents with equal access to health care services"
    docs = [
        "Search form. Health care is a basic human need. An effective health care system that spends wisely and covers everyone is critical for public health, safety and economic security. A single-payer health care system covers everyone. It has succeeded in countries throughout the developed world.",
        "Under a single-payer system, all residents of the U.S. would be covered for all medically necessary services, including doctor, hospital, preventive, long-term care, mental health, reproductive health care, dental, vision, prescription drug and medical supply costs.",
    ]

    # Based on https://github.com/terrierteam/pyterrier_t5/blob/main/pyterrier_t5/__init__.py#L44
    prompt = orig_tokenizer.batch_encode_plus(['Relevant:']*2, return_tensors='pt', padding='longest')
    max_vlen = orig_model.config.n_positions - prompt['input_ids'].shape[1]
    enc = orig_tokenizer.batch_encode_plus([f'Query: {q} Document: {d}' for q, d in zip([query,query], docs)], return_tensors='pt', padding='longest')
    for key, enc_value in list(enc.items()):
        enc_value = enc_value[:, :-1] # chop off end of sequence token-- this will be added with the prompt
        enc_value = enc_value[:, :max_vlen] # truncate any tokens that will not fit once the prompt is added
        enc[key] = torch.cat([enc_value, prompt[key][:enc_value.shape[0]]], dim=1) # add in the prompt to the end
    enc['decoder_input_ids'] = torch.full(
        (2, 1),
        orig_model.config.decoder_start_token_id,
        dtype=torch.long,
    )
    enc = {k: v for k, v in enc.items()}
    scores = [] 
    with torch.no_grad():
        result = orig_model(**enc).logits
    result = result[:, 0, (orig_tokenizer.encode('true')[0], orig_tokenizer.encode('false')[0])]
    scores = F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()

    with torch.inference_mode():
        output = module.score(queries=query, docs=docs)

    assert torch.allclose(output.scores, scores, atol=1e-4)
