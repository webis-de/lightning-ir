import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, T5EncoderModel

from lightning_ir.bi_encoder.tokenizer import BiEncoderTokenizer
from lightning_ir.models.xtr.model import XTRModel


class TestXTRModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained("google/xtr-base-en")
        self.linear = torch.nn.Linear(self.model.config.hidden_size, 128, bias=False)
        linear_layer_path = hf_hub_download("google/xtr-base-en", filename="2_Dense/pytorch_model.bin")
        state_dict = torch.load(linear_layer_path)
        state_dict["weight"] = state_dict.pop("linear.weight")
        self.linear.load_state_dict(state_dict)

    def forward(self, **kwargs):
        encoded = self.model(**kwargs).last_hidden_state
        embeddings = self.linear(encoded)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings


def test_same_as_xtr():
    model_name = "google/xtr-base-en"
    orig_model = TestXTRModel().eval()
    orig_tokenizer = AutoTokenizer.from_pretrained(model_name)

    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]
    orig_query_encoding = orig_tokenizer(query, return_tensors="pt")
    orig_doc_encoding = orig_tokenizer(documents, return_tensors="pt", padding=True, truncation=True)

    model = XTRModel.from_pretrained(model_name).eval()
    tokenizer = BiEncoderTokenizer.from_pretrained(model_name, config=model.config)
    query_encoding = tokenizer.tokenize_query(query, return_tensors="pt")
    doc_encoding = tokenizer.tokenize_doc(documents, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        query_embedding = orig_model(**orig_query_encoding)
        doc_embedding = orig_model(**orig_doc_encoding)
        output = model.forward(query_encoding=query_encoding, doc_encoding=doc_encoding)

    assert torch.allclose(query_embedding, output.query_embeddings.embeddings, atol=1e-6)
    assert torch.allclose(
        doc_embedding[orig_doc_encoding.attention_mask.bool()],
        output.doc_embeddings.embeddings[doc_encoding.attention_mask.bool()],
        atol=1e-6,
    )
