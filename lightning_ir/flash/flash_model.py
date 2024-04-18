from typing import Type, TypeVar

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from .flash_bert import FlashBertMixin
from .flash_electra import FlashElectraMixin
from .flash_mixin import FlashMixin
from .flash_roberta import FlashRobertaMixin


class AutoFlashModel(AutoModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model_class = AutoModel._model_mapping[type(config)]
        FlashModel = FlashClassFactory(model_class)
        return FlashModel.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


T = TypeVar("T", bound=PreTrainedModel)


def FlashClassFactory(TransformerModel: Type[T]) -> Type[T]:
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

    FlashClass = type(
        f"Flash{TransformerModel.__name__}",
        (Mixin, TransformerModel),
        {"__init__": __init__, "config_class": FlashConfig},
    )
    return FlashClass


def get_mixin(TransformerModel: Type[PreTrainedModel]) -> Type[FlashMixin]:
    if issubclass(TransformerModel, BertPreTrainedModel):
        return FlashBertMixin
    elif issubclass(TransformerModel, ElectraPreTrainedModel):
        return FlashElectraMixin
    elif issubclass(TransformerModel, RobertaPreTrainedModel):
        return FlashRobertaMixin
    else:
        raise ValueError(
            f"Model type {TransformerModel.__name__} not supported by Flash"
        )
