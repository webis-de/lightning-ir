from typing import Literal

from lightning_ir.models.mvr.config import MVRConfig
from lightning_ir import BiEncoderTokenizer
import torch
from transformers import BatchEncoding, AutoConfig, AutoModel, AutoTokenizer

from lightning_ir import BiEncoderModel, BiEncoderOutput

class MVRModel(BiEncoderModel):
    config = MVRConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)


AutoConfig.register(MVRConfig.model_type, MVRConfig)
AutoModel.register(MVRConfig, MVRModel)
AutoTokenizer.register(MVRConfig, BiEncoderTokenizer)