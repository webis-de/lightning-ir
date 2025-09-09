from pylate import indexes, models, retrieve
from lightning_ir.retrieve.plaid.plaid_indexer import PlaidIndexer, PlaidIndexConfig
from lightning_ir.retrieve.plaid.plaid_searcher import PlaidSearcher, PlaidSearchConfig
from lightning_ir import IndexBatch, SearchBatch, BiEncoderModule
from pathlib import Path
import numpy as np
import torch

from tests.conftest import CORPUS_DIR, DATA_DIR


def test_plaid():
    model_name = "colbert-ir/colbertv2.0"
    query = "What is the capital of France?"

    pylate_model = models.ColBERT(
        model_name_or_path=model_name,
    )

    pylate_index = indexes.PLAID(
        index_folder=Path(DATA_DIR) / "indexes/pylate-index",
        index_name="index",
        override=True,
    )

    documents_path = Path(CORPUS_DIR) / "docs.tsv"
    with documents_path.open("r", encoding="utf-8") as f:
        documents_ids, documents = zip(*[doc.strip().split("\t") for doc in f.readlines()])

    pylate_documents_embeddings = pylate_model.encode(
        documents,
        batch_size=32,
        is_query=False,
        show_progress_bar=False,
    )

    pylate_index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=pylate_documents_embeddings,
    )

    retriever = retrieve.ColBERT(index=pylate_index)

    pylate_queries_embeddings = pylate_model.encode(
        [query],
        batch_size=32,
        is_query=True,
        show_progress_bar=True,
    )

    pylate_scores = retriever.retrieve(
        queries_embeddings=pylate_queries_embeddings,
        k=10,  # Retrieve the top 10 matches for each query
    )

    module = BiEncoderModule(model_name).eval()
    index_batch = IndexBatch(doc_ids=documents_ids, docs=documents)
    with torch.inference_mode():
        output = module(index_batch)

    index_dir = Path(DATA_DIR) / "indexes/lightning-ir-index"
    index_config = PlaidIndexConfig(num_centroids=512)
    lightning_index = PlaidIndexer(
        index_dir=index_dir,
        index_config=index_config,
        module=module,
        verbose=True,
    )

    import faiss

    faiss.omp_set_num_threads(1)

    lightning_index.add(index_batch=index_batch, output=output)
    lightning_index.save()

    search_batch = SearchBatch(query_ids=["1"], queries=[query], doc_ids=documents_ids)
    search_config = PlaidSearchConfig(k=10)
    lightning_searcher = PlaidSearcher(
        index_dir=DATA_DIR / "indexes/lightning-ir-index", search_config=search_config, module=module
    )

    lightning_query_embedding = module(search_batch)
    lightning_scores = lightning_searcher.search(lightning_query_embedding)

    lightning_top_scores = lightning_scores[0]
    lightning_top_ids = lightning_scores[1][0]

    number_top_scores = len(lightning_top_scores)
    pylate_top_scores = np.array([doc["score"] for doc in pylate_scores[0][:number_top_scores]])

    pylate_top_ids = [doc["id"] for doc in pylate_scores[0][:number_top_scores]]
    assert pylate_top_ids == lightning_top_ids, "Top document IDs do not match between pylate and lightning_ir"
    assert np.allclose(
        pylate_top_scores, lightning_top_scores, atol=0.1
    ), "Top scores do not match closely between pylate and lightning_ir"


def test_plaid_indexing():
    model_name = "lightonai/GTE-ModernColBERT-v1"

    pylate_model = models.ColBERT(
        model_name_or_path=model_name,
    )

    pylate_index = indexes.PLAID(
        index_folder=Path(DATA_DIR) / "indexes/pylate-index",
        index_name="index",
        override=True,
    )

    documents_path = Path(CORPUS_DIR) / "docs.tsv"
    with documents_path.open("r", encoding="utf-8") as f:
        documents_ids, documents = zip(*[doc.strip().split("\t") for doc in f.readlines()])

    pylate_documents_embeddings = pylate_model.encode(
        documents,
        batch_size=32,
        is_query=False,
        show_progress_bar=False,
    )

    pylate_index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=pylate_documents_embeddings,
    )

    module = BiEncoderModule(model_name).eval()
    index_batch = IndexBatch(doc_ids=documents_ids, docs=documents)
    with torch.inference_mode():
        output = module(index_batch)

    index_dir = Path(DATA_DIR) / "indexes/lightning-ir-index"
    index_config = PlaidIndexConfig(num_centroids=512)
    lightning_index = PlaidIndexer(
        index_dir=index_dir,
        index_config=index_config,
        module=module,
        verbose=True,
    )

    import faiss

    faiss.omp_set_num_threads(1)

    lightning_index.add(index_batch=index_batch, output=output)
    lightning_index._train(force=True)

    pylate_index_path = Path(DATA_DIR) / "indexes/pylate-index/index"

    # Helper functions to load codes, residuals, centroids
    def load_pylate_codes(index_path):
        codes_path = index_path / "0.codes.pt"
        codes = torch.load(codes_path)
        return codes.numpy() if hasattr(codes, "numpy") else np.array(codes)

    def load_pylate_residuals(index_path):
        residuals_path = index_path / "0.residuals.pt"
        residuals = torch.load(residuals_path)
        return residuals.numpy() if hasattr(residuals, "numpy") else np.array(residuals)

    def load_pylate_centroids(index_path):
        centroids_path = index_path / "centroids.pt"
        centroids = torch.load(centroids_path)
        return centroids.numpy() if hasattr(centroids, "numpy") else np.array(centroids)

    pylate_codes = load_pylate_codes(pylate_index_path)
    pylate_residuals = load_pylate_residuals(pylate_index_path)
    pylate_centroids = load_pylate_centroids(pylate_index_path)

    lightning_codes = lightning_index.codes
    lightning_residuals = lightning_index.residuals
    lightning_centroids = lightning_index.residual_codec.centroids

    assert np.allclose(pylate_codes, lightning_codes), "Codes do not match between pylate and lightning_ir indices"
    assert np.allclose(
        pylate_residuals, lightning_residuals
    ), "Residuals do not match between pylate and lightning_ir indices"
    assert np.allclose(
        pylate_centroids, lightning_centroids
    ), "Centroids do not match between pylate and lightning_ir indices"
