# import torch
# from colbert.modeling.checkpoint import Checkpoint
# from colbert.modeling.colbert import colbert_score

# from lightning_ir.bi_encoder.col import ColModel
# from lightning_ir.bi_encoder.tokenizer import BiEncoderTokenizer


# def test_same_as_colbert():
#     query = "What is the capital of France?"
#     documents = [
#         "Paris is the capital of France.",
#         "France is a country in Europe.",
#         "The Eiffel Tower is in Paris.",
#     ]

#     model = ColModel.from_colbert_checkpoint("colbert-ir/colbertv2.0")
#     tokenizer = BiEncoderTokenizer.from_pretrained(
#         "colbert-ir/colbertv2.0", **model.config.to_dict()
#     )
#     query_encoding = tokenizer.tokenize_queries(query, return_tensors="pt")
#     doc_encoding = tokenizer.tokenize_docs(
#         documents, return_tensors="pt", padding=True, truncation=True
#     )
#     with torch.no_grad():
#         query_embedding = model.encode_queries(
#             query_encoding.input_ids, query_encoding.attention_mask
#         )
#         doc_embedding = model.encode_docs(
#             doc_encoding.input_ids, doc_encoding.attention_mask
#         )
#     query_scoring_mask, doc_scoring_mask = model.scoring_masks(
#         query_encoding.input_ids,
#         doc_encoding.input_ids,
#         query_encoding.attention_mask,
#         doc_encoding.attention_mask,
#     )
#     scores = model.score(
#         query_embedding,
#         doc_embedding,
#         query_scoring_mask,
#         doc_scoring_mask,
#         None,
#     )

#     orig_model = Checkpoint("colbert-ir/colbertv2.0")
#     orig_query = orig_model.queryFromText([query])
#     orig_docs = orig_model.docFromText(documents)
#     d_mask = ~(orig_docs == 0).all(-1)
#     orig_scores = colbert_score(orig_query, orig_docs, d_mask)

#     iterator = zip(
#         model.bert.state_dict().items(), orig_model.bert.state_dict().items()
#     )
#     for (key, weight), (orig_key, orig_weight) in iterator:
#         assert key == orig_key
#         if "word_embeddings" not in key:
#             assert torch.allclose(weight, orig_weight)

#     assert torch.allclose(query_embedding, orig_query)
#     assert torch.allclose(doc_embedding[doc_scoring_mask], orig_docs[d_mask])
#     assert torch.allclose(scores, orig_scores)
