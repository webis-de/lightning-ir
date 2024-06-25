from . import flash_bert

FLASH_ATTENTION_MAP = {
    "bert": (flash_bert.flash_attention_forward, flash_bert.SELF_ATTENTION_PATTERN),
    "roberta": (flash_bert.flash_attention_forward, flash_bert.SELF_ATTENTION_PATTERN),
    "electra": (flash_bert.flash_attention_forward, flash_bert.SELF_ATTENTION_PATTERN),
}
