from lightning_ir.tokenizer.tokenizer import BiEncoderTokenizer, CrossEncoderTokenizer


def test_bi_encoder_tokenizer(model_name_or_path: str):
    query = "What is the capital of France?"
    doc = "Paris is the capital of France."
    tokenizer = BiEncoderTokenizer.from_pretrained(model_name_or_path)
    query_encoding = tokenizer.tokenize_queries(query)
    doc_encoding = tokenizer.tokenize_docs(doc)
    assert query_encoding.input_ids[1] == tokenizer.query_token_id
    assert doc_encoding.input_ids[1] == tokenizer.doc_token_id
