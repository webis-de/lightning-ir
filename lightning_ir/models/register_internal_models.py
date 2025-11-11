from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..bi_encoder import BiEncoderConfig, BiEncoderTokenizer
from ..cross_encoder import CrossEncoderConfig, CrossEncoderTokenizer
from . import (
    CoilConfig,
    CoilModel,
    ColConfig,
    ColModel,
    ColTokenizer,
    DprConfig,
    DprModel,
    MonoConfig,
    MonoModel,
    MvrConfig,
    MvrModel,
    MvrTokenizer,
    SetEncoderConfig,
    SetEncoderModel,
    SetEncoderTokenizer,
    SpladeConfig,
    SpladeModel,
    SpladeTokenizer,
    UniCoilConfig,
    UniCoilModel,
    XTRConfig,
    XTRModel,
)


def _register_internal_models():
    AutoTokenizer.register(BiEncoderConfig, BiEncoderTokenizer)
    AutoTokenizer.register(CrossEncoderConfig, CrossEncoderTokenizer)
    AutoConfig.register(CoilConfig.model_type, CoilConfig)
    AutoModel.register(CoilConfig, CoilModel)
    AutoTokenizer.register(CoilConfig, BiEncoderTokenizer)
    AutoConfig.register(ColConfig.model_type, ColConfig)
    AutoModel.register(ColConfig, ColModel)
    AutoTokenizer.register(ColConfig, ColTokenizer)
    AutoConfig.register(DprConfig.model_type, DprConfig)
    AutoModel.register(DprConfig, DprModel)
    AutoTokenizer.register(DprConfig, BiEncoderTokenizer)
    AutoConfig.register(MonoConfig.model_type, MonoConfig)
    AutoModel.register(MonoConfig, MonoModel)
    AutoTokenizer.register(MonoConfig, CrossEncoderTokenizer)
    AutoConfig.register(MvrConfig.model_type, MvrConfig)
    AutoModel.register(MvrConfig, MvrModel)
    AutoTokenizer.register(MvrConfig, MvrTokenizer)
    AutoConfig.register(SetEncoderConfig.model_type, SetEncoderConfig)
    AutoModel.register(SetEncoderConfig, SetEncoderModel)
    AutoTokenizer.register(SetEncoderConfig, SetEncoderTokenizer)
    AutoConfig.register(SpladeConfig.model_type, SpladeConfig)
    AutoModel.register(SpladeConfig, SpladeModel)
    AutoTokenizer.register(SpladeConfig, SpladeTokenizer)
    AutoConfig.register(UniCoilConfig.model_type, UniCoilConfig)
    AutoModel.register(UniCoilConfig, UniCoilModel)
    AutoTokenizer.register(UniCoilConfig, BiEncoderTokenizer)
    AutoConfig.register(XTRConfig.model_type, XTRConfig)
    AutoModel.register(XTRConfig, XTRModel)
    AutoTokenizer.register(XTRConfig, ColTokenizer)
