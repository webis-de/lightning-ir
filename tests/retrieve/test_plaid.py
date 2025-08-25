from pylate import indexes, models
from lightning_ir.retrieve.plaid.plaid_indexer import PlaidIndexer, PlaidIndexConfig
from lightning_ir import IndexBatch, BiEncoderModule
from pathlib import Path
import torch

from tests.conftest import CORPUS_DIR, DATA_DIR


def test_plaid():
    model_name = "lightonai/GTE-ModernColBERT-v1"
    # Step 1: Load the ColBERT model
    model = models.ColBERT(
        model_name_or_path=model_name,
    )

    # Step 2: Initialize the PLAID index
    index = indexes.PLAID(
        index_folder=Path(DATA_DIR) / "indexes/pylate-index",
        index_name="index",
        override=True,
    )

    # Step 3: Encode the documents
    documents_path = Path(CORPUS_DIR) / "docs.tsv"
    with documents_path.open("r", encoding="utf-8") as f:
        documents_ids, documents = zip(*[doc.strip().split("\t") for doc in f.readlines()])

    documents_embeddings = model.encode(
        documents,
        batch_size=32,
        is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
        show_progress_bar=False,
    )

    # Step 4: Add document embeddings to the index by providing embeddings and corresponding ids
    index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=documents_embeddings,
    )

    module = BiEncoderModule(model_name).eval()
    index_batch = IndexBatch(doc_ids=documents_ids, docs=documents)
    with torch.inference_mode():
        output = module(index_batch)
    # Step 5: Initialize the lightning_ir PLAID index
    index_dir = Path(DATA_DIR) / "indexes/lightning-ir-index"
    index_config = PlaidIndexConfig(num_centroids=512, num_train_embeddings=1024)  # 512 centroids
    lightning_index = PlaidIndexer(
        index_dir=index_dir,
        index_config=index_config,
        module=module,
        verbose=True,
    )

    # Step 6: Add document embeddings to the lightning_ir PLAID index
    # import faiss

    # faiss.omp_set_num_threads(1)

    lightning_index.add(index_batch=index_batch, output=output)
    lightning_index._train(force=True)

    # Step 7: Compare the indices
    # assert lightning_index.num_embeddings == index, "Number of embeddings do not match"
    # assert torch.equal(index.codes, lightning_index.codes), "Codes do not match"
    # assert torch.equal(index.residuals, lightning_index.residuals), "Residuals do not match"
    # assert centroids
