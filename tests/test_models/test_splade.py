import pytest
import torch
from sentence_transformers import SparseEncoder
from transformers import BatchEncoding

from lightning_ir import BiEncoderModule
from lightning_ir.models import SpladeConfig


@pytest.mark.model
@pytest.mark.parametrize(
    "hf_model",
    [
        "naver/splade-v3",
        "naver/splade-v3-distilbert",
        "naver/splade-v3-doc",
        "naver/splade-v3-lexical",
        "naver/splade_v2_distil",
        "opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    ],
    indirect=True,
)
def test_same_as_splade(hf_model: str):
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]

    orig_model = SparseEncoder(hf_model)
    orig_query_embeddings = orig_model.encode_query([query])
    orig_doc_embeddings = orig_model.encode_document(documents)
    orig_scores = orig_model.similarity(orig_query_embeddings, orig_doc_embeddings)

    module = BiEncoderModule(hf_model).eval()
    with torch.inference_mode():
        output = module.score(query, documents)
    query_embeddings = output.query_embeddings
    doc_embeddings = output.doc_embeddings
    scores = output.scores

    assert torch.allclose(query_embeddings.embeddings.squeeze(1), orig_query_embeddings.to_dense().cpu(), atol=1e-4)
    assert torch.allclose(doc_embeddings.embeddings.squeeze(1), orig_doc_embeddings.to_dense().cpu(), atol=1e-4)
    assert torch.allclose(scores, orig_scores.cpu().view(-1), atol=1e-4)


def test_chunked_encode_matches_unchunked(model_name_or_path: str):
    """Chunked inference-time encoding must be equivalent to encoding the batch in one go.

    SPLADE builds a ``[batch, seq_len, vocab_size]`` intermediate, so a reranking batch -- which is
    one query's entire candidate pool -- can need hundreds of GiB. :meth:`SpladeModel.encode` splits
    it, and this pins down that splitting changes nothing but peak memory.
    """
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
        "Berlin is the capital of Germany.",
        "The Seine flows through Paris.",
    ]

    module = BiEncoderModule(
        model_name_or_path=model_name_or_path,
        config=SpladeConfig(query_length=8, doc_length=8),
    ).eval()

    # One module, toggled -- not two modules. The test backbone carries no pre-trained MLM head, so
    # two instances would get independently initialised projections and could not be compared.
    with torch.inference_mode():
        module.model.config.max_encode_chunk_elements = None
        unchunked = module.score(query, documents)
        module.model.config.max_encode_chunk_elements = 1
        chunked = module.score(query, documents)

    assert torch.allclose(chunked.doc_embeddings.embeddings, unchunked.doc_embeddings.embeddings, atol=1e-6)
    assert torch.allclose(chunked.query_embeddings.embeddings, unchunked.query_embeddings.embeddings, atol=1e-6)
    assert torch.allclose(chunked.scores, unchunked.scores, atol=1e-6)
    # The returned encoding must be the whole batch, not the final chunk.
    assert chunked.doc_embeddings.encoding["input_ids"].shape[0] == len(documents)


def test_encode_chunk_size(model_name_or_path: str):
    """The chunk size is derived from the budget, and never applies while gradients are enabled."""
    module = BiEncoderModule(
        model_name_or_path=model_name_or_path,
        config=SpladeConfig(query_length=8, doc_length=8),
    ).eval()
    model = module.model
    vocab_size = model.config.embedding_dim
    seq_len = 8
    encoding = BatchEncoding(
        {
            "input_ids": torch.zeros(16, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(16, seq_len, dtype=torch.long),
        }
    )

    with torch.inference_mode():
        # Budget for exactly four sequences.
        model.config.max_encode_chunk_elements = 4 * seq_len * vocab_size
        assert model._encode_chunk_size(encoding) == 4
        # A budget the whole batch fits inside must not chunk at all.
        model.config.max_encode_chunk_elements = 16 * seq_len * vocab_size
        assert model._encode_chunk_size(encoding) is None
        # Explicitly disabled.
        model.config.max_encode_chunk_elements = None
        assert model._encode_chunk_size(encoding) is None

    # Chunking under autograd would save no memory and would make training diverge from the
    # unchunked implementation, so it is never applied there however small the budget.
    model.config.max_encode_chunk_elements = 1
    with torch.enable_grad():
        assert model._encode_chunk_size(encoding) is None
