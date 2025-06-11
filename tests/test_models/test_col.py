import torch
import transformers

transformers.AdamW = None

from colbert.modeling.checkpoint import Checkpoint  # noqa: E402
from colbert.modeling.colbert import ColBERTConfig, colbert_score  # noqa: E402
from pylate import models, rank

from lightning_ir import BiEncoderModule  # noqa: E402


def test_same_as_colbert():
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    model_name = "colbert-ir/colbertv2.0"
    module = BiEncoderModule(model_name).eval()
    with torch.inference_mode():
        output = module.score(query, documents)
    query_embedding = output.query_embeddings
    doc_embedding = output.doc_embeddings

    colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(model_name))
    colbert_config.total_visible_gpus = 0
    orig_model = Checkpoint(model_name, colbert_config).cpu()
    orig_query = orig_model.queryFromText([query])
    orig_docs = orig_model.docFromText(documents)
    d_mask = ~(orig_docs == 0).all(-1)
    orig_scores = colbert_score(orig_query, orig_docs, d_mask, config=colbert_config)

    assert torch.allclose(query_embedding.embeddings, orig_query, atol=1e-6)
    assert torch.allclose(doc_embedding.embeddings[doc_embedding.scoring_mask], orig_docs[d_mask], atol=1e-6)
    assert torch.allclose(output.scores, orig_scores, atol=1e-6)


def test_same_as_modern_colbert():
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    model_name = "lightonai/GTE-ModernColBERT-v1"
    module = BiEncoderModule(model_name).eval()
    with torch.inference_mode():
        output = module.score(query, documents)
    query_embedding = output.query_embeddings
    doc_embedding = output.doc_embeddings

    orig_model = models.ColBERT(model_name_or_path=model_name)
    orig_query = orig_model.encode([query], is_query=True)
    orig_docs = orig_model.encode([documents], is_query=False)
    orig_scores = rank.rerank(
        queries_embeddings=orig_query, documents_embeddings=orig_docs, documents_ids=[list(range(len(documents)))]
    )

    assert torch.allclose(query_embedding.embeddings, torch.tensor(orig_query[0]), atol=1e-6)
    assert torch.allclose(
        doc_embedding.embeddings[doc_embedding.scoring_mask],
        torch.cat([torch.from_numpy(d) for doc in orig_docs for d in doc]),
        atol=1e-6,
    )
    assert torch.allclose(output.scores, torch.tensor([d["score"] for q in orig_scores for d in q]), atol=1e-6)
