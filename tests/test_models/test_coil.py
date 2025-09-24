import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from lightning_ir import BiEncoderModule

# COIL model implementation mostly from the repo below with minor simplications
# https://github.com/luyug/COIL/blob/6d15679f9ddb8f29c814d19e674333511c45feb3/modeling.py


class COIL(torch.nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)

        path = hf_hub_download(model_name_or_path, filename="model.pt")
        self.tok_proj = torch.nn.Linear(768, 32)
        self.cls_proj = torch.nn.Linear(768, 768)
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def forward(self, qry_input, doc_input):
        qry_out = self.model(**qry_input, return_dict=True)
        doc_out = self.model(**doc_input, return_dict=True)

        qry_cls = self.cls_proj(qry_out.last_hidden_state[:, 0])
        doc_cls = self.cls_proj(doc_out.last_hidden_state[:, 0])

        qry_reps = self.tok_proj(qry_out.last_hidden_state)  # Q * LQ * d
        doc_reps = self.tok_proj(doc_out.last_hidden_state)  # D * LD * d

        # mask ingredients
        doc_input_ids = doc_input["input_ids"]
        qry_input_ids = qry_input["input_ids"]
        qry_attention_mask = qry_input["attention_mask"]

        tok_scores = self.compute_tok_score_pair(doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask)

        # compute cls score separately
        cls_scores = (qry_cls * doc_cls).sum(-1)
        scores = tok_scores + cls_scores  # B

        # loss not defined during inference
        return scores.view(-1), qry_cls, doc_cls, qry_reps, doc_reps

    def compute_tok_score_pair(self, doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask):
        exact_match = qry_input_ids.unsqueeze(2) == doc_input_ids.unsqueeze(1)  # B * LQ * LD
        exact_match = exact_match.float()
        # qry_reps: B * LQ * d
        # doc_reps: B * LD * d
        scores_no_masking = torch.bmm(qry_reps, doc_reps.permute(0, 2, 1))  # B * LQ * LD
        tok_scores, _ = (scores_no_masking * exact_match).max(dim=2)  # B * LQ
        # remove padding and cls token
        tok_scores = (tok_scores * qry_attention_mask)[:, 1:].sum(-1)
        return tok_scores


@pytest.mark.parametrize("hf_model", ["fschlatt/coil-with-hn"], indirect=True)
def test_same_as_coil(hf_model: str):
    orig_model = COIL(hf_model).eval()
    orig_tokenizer = AutoTokenizer.from_pretrained(hf_model)

    query = "What is the capital of France?"
    docs = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    orig_query_encoded = orig_tokenizer(
        [query] * len(docs), padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    orig_docs_encoded = orig_tokenizer(docs, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.inference_mode():
        orig_scores, orig_qry_cls, orig_doc_cls, orig_qry_reps, orig_doc_reps = orig_model(
            orig_query_encoded, orig_docs_encoded
        )
    orig_qry_reps = orig_qry_reps[:, 1:]
    orig_doc_reps = orig_doc_reps[:, 1:]

    module = BiEncoderModule(model_name_or_path=hf_model).eval()
    output = module.score(query, docs)

    assert torch.allclose(output.query_embeddings.cls_embeddings[0].expand_as(orig_qry_cls), orig_qry_cls, atol=1e-4)
    assert torch.allclose(output.doc_embeddings.cls_embeddings.view_as(orig_doc_cls), orig_doc_cls, atol=1e-4)
    assert torch.allclose(
        output.query_embeddings.token_embeddings[:, : orig_qry_reps.shape[1]].expand_as(orig_qry_reps),
        orig_qry_reps,
        atol=1e-4,
    )
    assert torch.allclose(
        output.doc_embeddings.token_embeddings[:, : orig_doc_reps.shape[1]].view_as(orig_doc_reps),
        orig_doc_reps,
        atol=1e-4,
    )
    assert torch.allclose(output.scores, orig_scores, atol=1e-4)
