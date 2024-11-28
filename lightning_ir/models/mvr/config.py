from lightning_ir.bi_encoder.config import BiEncoderConfig

class MVRConfig(BiEncoderConfig):
    model_type = "mvr"

    ADDED_ARGS = BiEncoderConfig.ADDED_ARGS.union({"additional_linear_layer"})

    def __init__(self, 
                 add_viewer_tokens = True, 
                 num_viewer_tokens = 8,
                 **kwargs):
        super().__init__(**kwargs)
        self.add_viewer_tokens = add_viewer_tokens   
        self.num_viewer_tokens = num_viewer_tokens     