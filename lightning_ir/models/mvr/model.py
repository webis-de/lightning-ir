from dataclasses import dataclass
from typing import Literal

import torch

from lightning_ir.models.mvr.config import MVRConfig
from ...bi_encoder import BiEncoderModel, BiEncoderOutput


@dataclass
class MVROutput(BiEncoderOutput):
    """Dataclass containing the output of a MVR model."""

    viewer_token_scores: torch.tensor = None
    """individual similarity scores for each viewer token with query"""

class MVRModel(BiEncoderModel):
    config_class = MVRConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
