from pathlib import Path
import torch

from lightning_ir import BiEncoderModule, IndexBatch, SearchBatch
from lightning_ir.retrieve.plaid.plaid_indexer_fastplaid import PlaidIndexConfig, PlaidIndexerFastPlaid
from lightning_ir.retrieve.plaid.plaid_searcher_fastplaid import PlaidSearchConfig, PlaidSearcherFastPlaid
from tests.conftest import CORPUS_DIR, DATA_DIR


def test_plaid_fastplaid():
    lightning_index = PlaidIndexerFastPlaid(
        index_dir=Path(DATA_DIR) / "indexes" / "lightning-ir-plaid-fastplaid",
        index_config=PlaidIndexConfig(
            num_centroids=256,
            num_train_embeddings=1000,
            k_means_iters=4,
            n_bits=2,
            seed=42,
        ),
        module=BiEncoderModule(model_name_or_path="colbert-ir/colbertv2.0"),
        verbose=True,
    )

    documents_path = Path(CORPUS_DIR) / "docs.tsv"
    with documents_path.open("r", encoding="utf-8") as f:
        documents_ids, documents = zip(*[doc.strip().split("\t") for doc in f.readlines()])  # type: ignore
    documents_ids = list(documents_ids)
    documents = list(documents)

    output = lightning_index.module(IndexBatch(docs=documents, doc_ids=documents_ids))

    lightning_index.add(
        IndexBatch(docs=documents, doc_ids=documents_ids),
        output,
    )

    lightning_searcher = PlaidSearcherFastPlaid(
        index_dir=Path(DATA_DIR) / "indexes" / "lightning-ir-plaid-fastplaid",
        search_config=PlaidSearchConfig(k=10),
        module=lightning_index.module,
        use_gpu=torch.cuda.is_available(),
    )

    lightning_searcher.load()

    output = lightning_index.module(SearchBatch(queries=["What is the capital of France?"], query_ids=["q1"]))

    scores = lightning_searcher.search(output)

    assert scores is not None
