from abc import ABC, abstractmethod
from functools import partial

from transformers import PreTrainedModel


class FlashMixin(PreTrainedModel, ABC):
    encoder_name: str
    self_attention_pattern: str

    ADDITIONAL_KWARGS = []

    def __init__(self):
        for name, module in self.named_modules():
            if name.endswith(self.self_attention_pattern):
                module.forward = partial(self.flash_attention_forward, module)

    @staticmethod
    @abstractmethod
    def flash_attention_forward(*args, **kwargs):
        ...
