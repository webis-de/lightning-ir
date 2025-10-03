from pathlib import Path

import torch
from fast_plaid import search

from lightning_ir import BiEncoderModule, IndexBatch, SearchBatch
from lightning_ir.retrieve.plaid.plaid_indexer import PlaidIndexConfig, PlaidIndexer
from lightning_ir.retrieve.plaid.plaid_searcher import PlaidSearchConfig, PlaidSearcher
from tests.conftest import CORPUS_DIR, DATA_DIR


def test_plaid():
    query = "What is the capital of France?"
    module = BiEncoderModule(model_name_or_path="colbert-ir/colbertv2.0")
    index_config = PlaidIndexConfig(
        num_centroids=256,
        num_train_embeddings=1000,
        k_means_iters=4,
        n_bits=2,
        seed=42,
    )

    lightning_index = PlaidIndexer(
        index_dir=Path(DATA_DIR) / "indexes" / "lightning-ir-plaid",
        index_config=index_config,
        module=module,
        verbose=True,
    )

    documents_path = Path(CORPUS_DIR) / "docs.tsv"
    with documents_path.open("r", encoding="utf-8") as f:
        documents_ids, documents = zip(*[doc.strip().split("\t") for doc in f.readlines()])  # type: ignore
    documents_ids = list(documents_ids)
    documents = list(documents)

    doc_output = lightning_index.module(IndexBatch(docs=documents, doc_ids=documents_ids))

    lightning_index.add(
        IndexBatch(docs=documents, doc_ids=documents_ids),
        doc_output,
    )
    lightning_index.finalize()

    lightning_searcher = PlaidSearcher(
        index_dir=Path(DATA_DIR) / "indexes" / "lightning-ir-plaid",
        search_config=PlaidSearchConfig(k=10),
        module=lightning_index.module,
        use_gpu=torch.cuda.is_available(),
    )

    query_output = lightning_index.module(SearchBatch(queries=[query], query_ids=["q1"]))

    scores = lightning_searcher.search(query_output)

    assert scores is not None

    fast_plaid = search.FastPlaid(index="fast_plaid_index")

    fast_plaid.create(
        documents_embeddings=module(
            IndexBatch(docs=documents, doc_ids=documents_ids)
        ).doc_embeddings.embeddings.detach(),
        seed=index_config.seed,
        kmeans_niters=index_config.k_means_iters,
        nbits=index_config.n_bits,
        n_samples_kmeans=index_config.num_train_embeddings,
    )

    fast_plaid_scores = fast_plaid.search(
        queries_embeddings=module(SearchBatch(queries=[query], query_ids=["q1"])).query_embeddings.embeddings.detach(),
        top_k=10,
    )

    fast_plaid_doc_ids, fast_plaid_scores = zip(*fast_plaid_scores[0])

    assert fast_plaid_scores is not None
    assert scores[1] == list(fast_plaid_doc_ids)
    assert torch.allclose(torch.tensor(scores[0]), torch.tensor(fast_plaid_scores), atol=1e-4)
