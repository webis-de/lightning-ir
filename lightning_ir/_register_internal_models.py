from transformers import AutoConfig, AutoModel, AutoTokenizer

from .bi_encoder import BiEncoderConfig, BiEncoderModel, BiEncoderTokenizer
from .cross_encoder import CrossEncoderConfig, CrossEncoderModel, CrossEncoderTokenizer
from .models import (
    ColConfig,
    ColModel,
    SetEncoderConfig,
    SetEncoderModel,
    SetEncoderTokenizer,
    SpladeConfig,
    SpladeModel,
    T5CrossEncoderConfig,
    T5CrossEncoderModel,
    T5CrossEncoderTokenizer,
    XTRConfig,
    XTRModel,
)


def _register_internal_models():
    AutoConfig.register(BiEncoderConfig.model_type, BiEncoderConfig)
    AutoModel.register(BiEncoderConfig, BiEncoderModel)
    AutoTokenizer.register(BiEncoderConfig, BiEncoderTokenizer)
    AutoConfig.register(CrossEncoderConfig.model_type, CrossEncoderConfig)
    AutoModel.register(CrossEncoderConfig, CrossEncoderModel)
    AutoTokenizer.register(CrossEncoderConfig, CrossEncoderTokenizer)
    AutoConfig.register(ColConfig.model_type, ColConfig)
    AutoModel.register(ColConfig, ColModel)
    AutoTokenizer.register(ColConfig, BiEncoderTokenizer)
    AutoConfig.register(SetEncoderConfig.model_type, SetEncoderConfig)
    AutoModel.register(SetEncoderConfig, SetEncoderModel)
    AutoTokenizer.register(SetEncoderConfig, SetEncoderTokenizer)
    AutoConfig.register(SpladeConfig.model_type, SpladeConfig)
    AutoModel.register(SpladeConfig, SpladeModel)
    AutoTokenizer.register(SpladeConfig, BiEncoderTokenizer)
    AutoConfig.register(T5CrossEncoderConfig.model_type, T5CrossEncoderConfig)
    AutoModel.register(T5CrossEncoderConfig, T5CrossEncoderModel)
    AutoTokenizer.register(T5CrossEncoderConfig, T5CrossEncoderTokenizer)
    AutoConfig.register(XTRConfig.model_type, XTRConfig)
    AutoModel.register(XTRConfig, XTRModel)
    AutoTokenizer.register(XTRConfig, BiEncoderTokenizer)
