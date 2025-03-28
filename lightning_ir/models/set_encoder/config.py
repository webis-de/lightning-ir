from ...cross_encoder.config import CrossEncoderConfig


class SetEncoderConfig(CrossEncoderConfig):
    model_type = "set-encoder"

    ADDED_ARGS = CrossEncoderConfig.ADDED_ARGS.union({"depth", "add_extra_token", "sample_missing_docs"})
    TOKENIZER_ARGS = CrossEncoderConfig.TOKENIZER_ARGS.union({"add_extra_token"})

    def __init__(
        self,
        *args,
        depth: int = 100,
        add_extra_token: bool = False,
        sample_missing_docs: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.add_extra_token = add_extra_token
        self.sample_missing_docs = sample_missing_docs
