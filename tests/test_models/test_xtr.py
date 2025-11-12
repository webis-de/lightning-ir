import pytest
import torch
from transformers import AutoTokenizer
from xtr.configuration_xtr import XtrConfig
from xtr.modeling_xtr import XtrModel

from lightning_ir import BiEncoderModule


@pytest.mark.parametrize("hf_model", ["google/xtr-base-en"], indirect=True)
def test_same_as_xtr(hf_model: str):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    xtr_config = XtrConfig(hf_model)
    xtr_model = XtrModel(model_name_or_path=hf_model, config=xtr_config).eval()
    xtr_tokenizer = AutoTokenizer.from_pretrained(hf_model)

    query_inputs = xtr_tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    doc_inputs = xtr_tokenizer(documents, padding=True, truncation=True, return_tensors="pt")

    with torch.inference_mode():
        orig_query_embeddings = xtr_model(
            input_ids=query_inputs["input_ids"], attention_mask=query_inputs["attention_mask"]
        )
        orig_doc_embeddings = xtr_model(input_ids=doc_inputs["input_ids"], attention_mask=doc_inputs["attention_mask"])

    orig_query_mask = query_inputs["attention_mask"].bool()
    orig_doc_mask = doc_inputs["attention_mask"].bool()

    orig_query_flat = orig_query_embeddings[orig_query_mask]
    orig_doc_flat = orig_doc_embeddings[orig_doc_mask]
    orig_token_scores = torch.matmul(orig_query_embeddings.unsqueeze(1), orig_doc_embeddings.transpose(1, 2))
    orig_scores = orig_token_scores.max(dim=-1).values.sum(dim=-1).squeeze(0)
    orig_scores = orig_scores / orig_query_embeddings.shape[1]

    module = BiEncoderModule(hf_model).eval()
    with torch.inference_mode():
        output = module.score(query, documents)

    lightning_query_embeddings = output.query_embeddings.embeddings
    lightning_doc_embeddings = output.doc_embeddings.embeddings
    lightning_query_mask = output.query_embeddings.scoring_mask
    lightning_doc_mask = output.doc_embeddings.scoring_mask
    lightning_scores = output.scores

    lightning_query_flat = lightning_query_embeddings[lightning_query_mask]
    lightning_doc_flat = lightning_doc_embeddings[lightning_doc_mask]

    assert torch.allclose(lightning_query_flat, orig_query_flat, atol=1e-5), (
        f"Query embeddings differ. LightningIR shape: {lightning_query_flat.shape}, "
        f"XTR-pytorch shape: {orig_query_flat.shape}"
    )

    assert torch.allclose(lightning_doc_flat, orig_doc_flat, atol=1e-5), (
        f"Document embeddings differ. LightningIR shape: {lightning_doc_flat.shape}, "
        f"XTR-pytorch shape: {orig_doc_flat.shape}"
    )

    lightning_scores_sorted, _ = torch.sort(lightning_scores, descending=True)
    orig_scores_sorted, _ = torch.sort(orig_scores, descending=True)

    assert torch.allclose(lightning_scores_sorted, orig_scores_sorted, atol=1e-5), "Scores differ"


def test_xtr_training():
    model_name = "google/xtr-base-en"
    module = BiEncoderModule(model_name_or_path=model_name).train()

    queries = ["What is the capital of France?", "Where is the Eiffel Tower located?"]
    documents = [
        [
            "Paris is the capital of France.",
            "France is a country in Europe.",
            "The Eiffel Tower is in Paris.",
        ],
        [
            "The Eiffel Tower is located in Paris.",
            "London is the capital of the UK.",
        ],
    ]

    output = module.score(queries, documents)

    module.eval()

    output_eval = module.score(queries, documents)

    assert not torch.allclose(output.scores, output_eval.scores), "Scores should differ between training and eval"
