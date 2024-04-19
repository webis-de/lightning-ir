from lightning_ir.tokenizer.tokenizer import BiEncoderTokenizer, CrossEncoderTokenizer
from lightning_ir.bi_encoder.model import BiEncoderConfig
from lightning_ir.cross_encoder.model import CrossEncoderConfig
import pytest


@pytest.mark.parametrize(
    "config",
    (BiEncoderConfig(query_expansion=True, doc_expansion=True), CrossEncoderConfig()),
)
def test_serialize_deserialize(
    model_name_or_path: str,
    config: BiEncoderConfig | CrossEncoderConfig,
    tmpdir_factory: pytest.TempdirFactory,
):
    save_dir = tmpdir_factory.mktemp(config.model_type)
    tokenizer_kwargs = config.to_tokenizer_dict()
    if config.model_type == "bi-encoder":
        Tokenizer = BiEncoderTokenizer
    elif config.model_type == "cross-encoder":
        Tokenizer = CrossEncoderTokenizer
    else:
        raise ValueError("Invalid model type")
    tokenizer = Tokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    tokenizer.save_pretrained(save_dir)
    new_tokenizer = Tokenizer.from_pretrained(save_dir)
    assert tokenizer is not None
    assert new_tokenizer is not None
    for key in tokenizer_kwargs:
        assert getattr(tokenizer, key) == getattr(new_tokenizer, key)


def test_bi_encoder_tokenizer(model_name_or_path: str):
    query = "What is the capital of France?"
    doc = "Paris is the capital of France."
    tokenizer = BiEncoderTokenizer.from_pretrained(model_name_or_path)
    encoding = tokenizer.tokenize(query, doc)
    assert encoding is not None
    query_encoding = encoding["query_encoding"]
    doc_encoding = encoding["doc_encoding"]
    assert query_encoding.input_ids[1] == tokenizer.query_token_id
    assert doc_encoding.input_ids[1] == tokenizer.doc_token_id
