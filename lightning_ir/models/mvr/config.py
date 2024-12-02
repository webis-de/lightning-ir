from lightning_ir.bi_encoder.config import BiEncoderConfig

class MVRConfig(BiEncoderConfig):
    model_type = "mvr"

    TOKENIZER_ARGS = BiEncoderConfig.TOKENIZER_ARGS.union({
        "num_viewer_tokens",
    })

    def __init__(self, 
                 num_viewer_tokens: int | None = 8,
                 **kwargs):
        super().__init__(**kwargs)   
        self.num_viewer_tokens = num_viewer_tokens