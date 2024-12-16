from lightning_ir.bi_encoder.config import BiEncoderConfig


class MVRConfig(BiEncoderConfig):
    model_type = "mvr"

    TOKENIZER_ARGS = BiEncoderConfig.TOKENIZER_ARGS.union({"num_viewer_tokens", "add_marker_tokens"})

    def __init__(self, num_viewer_tokens: int | None = 8, add_marker_tokens: bool = False):
        super().__init__()
        self.num_viewer_tokens = num_viewer_tokens
        self.add_marker_tokens = add_marker_tokens
