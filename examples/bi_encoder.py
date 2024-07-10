from pathlib import Path

import torch

from lightning_ir import (
    BiEncoderConfig,
    BiEncoderModule,
    FaissFlatIndexConfig,
    FaissFlatIndexer,
    IndexBatch,
    SearchBatch,
    SearchConfig,
    Searcher,
)


def main():

    # 1. Load a pre-trained model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    index_dir = Path("./index")
    config = BiEncoderConfig(
        linear=False,
        embedding_dim=384,
        add_marker_tokens=False,
        similarity_function="cosine",
    )
    model = BiEncoderModule(model_name, config).eval()

    # 2. Create an index
    indexer = FaissFlatIndexer(
        index_dir, FaissFlatIndexConfig(model.config.similarity_function), model.config
    )
    docs = {
        "doc_id_1": "Paris is the capital of France.",
        "doc_id_2": "The Eiffel Tower is in Paris.",
        "doc_id_3": "France is in Europe.",
        "doc_id_4": "The Louvre is in Paris.",
        "doc_id_5": "The Seine river runs through Paris.",
        "doc_id_6": "Berlin is the capital of Germany.",
        "doc_id_7": "Germany is in Europe.",
        "doc_id_8": "The Brandenburg Gate is in Berlin.",
        "doc_id_9": "The Berlin Wall is in Berlin.",
        "doc_id_10": "The Rhine river runs through Germany.",
    }

    # 3. Embed the documents
    # for larger collections make sure to batch the documents
    index_batch = IndexBatch(
        doc_ids=tuple(docs.keys()), docs=tuple(tuple(docs.values()))
    )
    with torch.inference_mode():
        output = model.forward(index_batch)

    # 4. Add embeddings to index
    indexer.add(index_batch, output)
    indexer.save()

    # 5. Search the index
    searcher = Searcher(SearchConfig(Path("./index"), k=10_000, candidate_k=3), model)
    queries = {
        "query_1": "What is the capital of France?",
        "query_2": "What is the capital of Germany?",
        "query_3": "What is the power house the cell?",
    }
    search_batch = SearchBatch(
        query_ids=tuple(queries.keys()), queries=tuple(queries.values())
    )

    # 6. Embed the queries
    with torch.inference_mode():
        output = model.forward(search_batch)

    # 7. Search the index
    doc_scores, doc_ids, num_docs = searcher.search(output)
    doc_idx = 0
    for query, n in zip(queries.values(), num_docs):
        print(f"Query: {query}")
        for _ in range(n):
            score = doc_scores[doc_idx]
            doc_id = doc_ids[doc_idx]
            doc = docs[doc_id]
            print(f"Doc ({doc_id}) [{score:.4f}]: {doc}")
            doc_idx += 1


if __name__ == "__main__":
    main()
