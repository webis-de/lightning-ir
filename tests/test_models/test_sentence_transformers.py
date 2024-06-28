import torch
from sentence_transformers import SentenceTransformer

from lightning_ir import BiEncoderModel


def test_same_as_sentence_transformer():
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = BiEncoderModel.from_pretrained(
        model_name, add_marker_tokens=False, projection=None, embedding_dim=384
    )
    tokenizer = BiEncoderModel.config_class.tokenizer_class.from_pretrained(
        model_name, **model.config.to_tokenizer_dict()
    )
    encoding = tokenizer.tokenize_doc(sentences, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.encode_doc(**encoding)

    orig_model = SentenceTransformer(model_name)
    orig_embeddings = orig_model.encode(sentences)

    assert torch.allclose(
        embeddings.embeddings.squeeze(1), torch.from_numpy(orig_embeddings), atol=1e-6
    )
