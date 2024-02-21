from .flash_bert import FlashBertMixin


class FlashElectraMixin(FlashBertMixin):
    encoder_name = "electra"
