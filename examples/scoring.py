from lightning_ir import BiEncoderModule, CrossEncoderModule

bi_encoder = BiEncoderModule("webis/bert-bi-encoder")
cross_encoder = CrossEncoderModule("webis/monoelectra-base")

query = "What is the capital of France?"
docs = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
bi_encoder_output = bi_encoder.score(query, docs)
cross_encoder_output = cross_encoder.score(query, docs)

print(bi_encoder_output.scores)
# tensor([38.9621, 29.7557])
print(bi_encoder_output.query_embeddings.embeddings.shape)
# torch.Size([1, 1, 768])
print(bi_encoder_output.doc_embeddings.embeddings.shape)
# torch.Size([2, 1, 768])
print(cross_encoder_output.scores)
# tensor([ 7.7892, -3.5815])
print(cross_encoder_output.embeddings.shape)
# torch.Size([2, 1, 768])
