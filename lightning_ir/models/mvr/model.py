from typing import Literal

from lightning_ir.models.mvr.config import MVRConfig

from ...bi_encoder import BiEncoderModel

class MVRModel(BiEncoderModel):
    config_class = MVRConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)


