from lightning_ir.bi_encoder.config import BiEncoderConfig

class MVRConfig(BiEncoderConfig):
    model_type = "mvr"

    TOKENIZER_ARGS = BiEncoderConfig.TOKENIZER_ARGS.union({
        "add_viewer_tokens",
        "num_viewer_tokens",
    })

    def __init__(self, 
                 add_viewer_tokens = True, 
                 num_viewer_tokens = 8,
                 **kwargs):
        super().__init__(**kwargs)
        self.add_viewer_tokens = add_viewer_tokens   
        self.num_viewer_tokens = num_viewer_tokens