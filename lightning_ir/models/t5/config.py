from typing import Literal

from ...cross_encoder.config import CrossEncoderConfig


class T5CrossEncoderConfig(CrossEncoderConfig):

    model_type = "encoder-decoder-cross-encoder"

    TOKENIZER_ARGS = CrossEncoderConfig.TOKENIZER_ARGS.union({"decoder_strategy"})
    ADDED_ARGS = CrossEncoderConfig.ADDED_ARGS.union(TOKENIZER_ARGS)

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        decoder_strategy: Literal["mono", "rank"] = "mono",
        **kwargs,
    ) -> None:
        kwargs["pooling_strategy"] = "first"
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.decoder_strategy = decoder_strategy
