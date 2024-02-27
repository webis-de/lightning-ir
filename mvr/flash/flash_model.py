from typing import Type

from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel

from .flash_bert import FlashBertMixin
from .flash_electra import FlashElectraMixin
from .flash_mixin import FlashMixin


class AutoFlashModel(AutoModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model_class = AutoModel._model_mapping[type(config)]
        FlashModel = FlashClassFactory(model_class)
        return FlashModel.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


def FlashClassFactory(
    TransformerModel: Type[PreTrainedModel],
) -> Type[PreTrainedModel]:
    Mixin = get_mixin(TransformerModel)

    assert issubclass(TransformerModel.config_class, PretrainedConfig)
    FlashConfig = type(
        f"Flash{TransformerModel.config_class.__name__}",
        (TransformerModel.config_class,),
        {},
    )

    def __init__(self, config: PretrainedConfig) -> None:
        TransformerModel.__init__(self, config)
        Mixin.__init__(self)

    flash_class = type(
        f"Flash{TransformerModel.__name__}",
        (Mixin, TransformerModel),
        {"__init__": __init__, "config_class": FlashConfig},
    )
    return flash_class


def get_mixin(TransformerModel: Type[PreTrainedModel]) -> Type[FlashMixin]:
    if issubclass(TransformerModel, BertPreTrainedModel):
        return FlashBertMixin
    elif issubclass(TransformerModel, ElectraPreTrainedModel):
        return FlashElectraMixin
    else:
        raise ValueError(
            f"Model type {TransformerModel.__name__} not supported by Flash"
        )
