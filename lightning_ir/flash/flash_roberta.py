from .flash_bert import FlashBertMixin


class FlashRobertaMixin(FlashBertMixin):
    encoder_name = "roberta"
