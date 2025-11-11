import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, PreTrainedModel

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


# UniCOIL model implementation mostly from the repo below with minor simplications
# https://github.com/castorini/pyserini/blob/master/pyserini/encode/_unicoil.py


class UniCoilEncoder(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "coil_encoder"
    load_tf_weights = None

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.bert.init_weights()
        self.tok_proj.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.bert.config.pad_token_id)
            )
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        tok_weights = self.tok_proj(sequence_output)[..., 0]
        tok_weights = torch.relu(tok_weights)
        tok_weights = tok_weights.masked_fill(~(attention_mask.bool()), 0)
        out = torch.zeros(input_shape[0], self.config.vocab_size, device=device)
        out = out.scatter(1, input_ids, tok_weights)
        return out


@pytest.mark.model
@pytest.mark.parametrize(
    "hf_model",
    [
        "fschlatt/coil-with-hn",
    ],
    indirect=True,
)
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


@pytest.mark.model
@pytest.mark.parametrize(
    "hf_model",
    [
        "castorini/unicoil-noexp-msmarco-passage",
    ],
    indirect=True,
)
def test_same_as_unicoil(hf_model: str):
    query = "What is the capital of France?"
    docs = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    orig_model = UniCoilEncoder.from_pretrained(hf_model)
    orig_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    orig_query_encoded = orig_tokenizer(
        query, padding=True, truncation=True, max_length=512, return_tensors="pt", return_token_type_ids=False
    )
    orig_docs_encoded = orig_tokenizer(
        docs, padding=True, truncation=True, max_length=512, return_tensors="pt", return_token_type_ids=False
    )
    with torch.inference_mode():
        orig_query = orig_model(**orig_query_encoded)
        orig_docs = orig_model(**orig_docs_encoded)
        orig_scores = torch.matmul(orig_query[:, None], orig_docs[..., None]).squeeze(-1).view(-1)

    module = BiEncoderModule(model_name_or_path=hf_model).eval()
    output = module.score(query, docs)

    assert torch.allclose(output.query_embeddings.embeddings.view_as(orig_query), orig_query, atol=1e-4)
    assert torch.allclose(output.doc_embeddings.embeddings.view_as(orig_docs), orig_docs, atol=1e-4)
    assert torch.allclose(output.scores, orig_scores, atol=1e-4)
