import torch
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import ColBERTConfig, colbert_score

from lightning_ir import BiEncoderTokenizer, ColModel


def test_same_as_colbert():
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    model_name = "colbert-ir/colbertv2.0"
    model = ColModel.from_colbert_checkpoint(model_name).eval()
    tokenizer = BiEncoderTokenizer.from_pretrained(model_name, **model.config.to_dict())
    query_encoding = tokenizer.tokenize_query(query, return_tensors="pt")
    doc_encoding = tokenizer.tokenize_doc(documents, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        query_embedding = model.encode_query(query_encoding.input_ids, query_encoding.attention_mask)
        doc_embedding = model.encode_doc(doc_encoding.input_ids, doc_encoding.attention_mask)
    scores = model.score(query_embedding, doc_embedding, None)

    colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(model_name))
    orig_model = Checkpoint(model_name, colbert_config)
    orig_query = orig_model.queryFromText([query])
    orig_docs = orig_model.docFromText(documents)
    d_mask = ~(orig_docs == 0).all(-1)
    orig_scores = colbert_score(orig_query, orig_docs, d_mask)

    assert torch.allclose(query_embedding.embeddings, orig_query, atol=1e-7)
    assert torch.allclose(doc_embedding.embeddings[doc_embedding.scoring_mask], orig_docs[d_mask], atol=1e-7)
    assert torch.allclose(scores, orig_scores, atol=1e-7)
