from lightning_ir.bi_encoder.config import BiEncoderConfig

class MVRConfig(BiEncoderConfig):
    model_type = "mvr"

    ADDED_ARGS = BiEncoderConfig.ADDED_ARGS.union({"additional_linear_layer"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)