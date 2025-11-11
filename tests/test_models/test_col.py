import pytest
import torch
import transformers

transformers.AdamW = None  # monkey patch to avoid import error for original colbert

from colbert.modeling.checkpoint import Checkpoint  # noqa: E402
from colbert.modeling.colbert import ColBERTConfig, colbert_score  # noqa: E402
from pylate import models, rank  # noqa: E402
from pylate.models.colbert import ColBERT as PyLateColbert  # noqa: E402

from lightning_ir import BiEncoderModule  # noqa: E402


# monkeypatch pylate colbert for newest transformers compatibility
def _get_model_type(*args, **kwargs):
    return "ColBERT"


PyLateColbert._get_model_type = _get_model_type


@pytest.mark.model
@pytest.mark.parametrize(
    "hf_model",
    [
        "colbert-ir/colbertv2.0",
    ],
    indirect=True,
)
def test_same_as_colbert(hf_model: str):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    module = BiEncoderModule(hf_model).eval()
    with torch.inference_mode():
        output = module.score(query, documents)
    query_embedding = output.query_embeddings
    doc_embedding = output.doc_embeddings

    colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(hf_model))
    colbert_config.total_visible_gpus = 0
    orig_model = Checkpoint(hf_model, colbert_config).cpu()
    orig_query = orig_model.queryFromText([query])
    orig_docs = orig_model.docFromText(documents)
    d_mask = ~(orig_docs == 0).all(-1)
    orig_scores = colbert_score(orig_query, orig_docs, d_mask, config=colbert_config)

    assert torch.allclose(query_embedding.embeddings, orig_query, atol=1e-6)
    assert torch.allclose(doc_embedding.embeddings[doc_embedding.scoring_mask], orig_docs[d_mask], atol=1e-6)
    assert torch.allclose(output.scores, orig_scores, atol=1e-6)


@pytest.mark.model
@pytest.mark.parametrize(
    "hf_model",
    [
        "lightonai/GTE-ModernColBERT-v1",
    ],
    indirect=True,
)
def test_same_as_modern_colbert(hf_model: str):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    hf_model = "lightonai/GTE-ModernColBERT-v1"
    module = BiEncoderModule(hf_model).eval()
    with torch.inference_mode():
        output = module.score(query, documents)
    query_embedding = output.query_embeddings
    doc_embedding = output.doc_embeddings

    orig_model = models.ColBERT(model_name_or_path=hf_model)
    orig_query = orig_model.encode([query], is_query=True)
    orig_docs = orig_model.encode([documents], is_query=False)
    orig_scores = rank.rerank(
        queries_embeddings=orig_query, documents_embeddings=orig_docs, documents_ids=[list(range(len(documents)))]
    )

    assert torch.allclose(
        query_embedding.embeddings[query_embedding.scoring_mask], torch.tensor(orig_query[0]), atol=1e-6
    )
    assert torch.allclose(
        doc_embedding.embeddings[doc_embedding.scoring_mask],
        torch.cat([torch.from_numpy(d) for doc in orig_docs for d in doc]),
        atol=1e-6,
    )
    assert torch.allclose(output.scores, torch.tensor([d["score"] for q in orig_scores for d in q]), atol=1e-6)
