from transformers import AutoConfig, AutoModel, AutoTokenizer

from .bi_encoder import BiEncoderConfig, BiEncoderTokenizer
from .cross_encoder import CrossEncoderConfig, CrossEncoderModel, CrossEncoderTokenizer
from .models import (
    ColConfig,
    ColModel,
    ColTokenizer,
    DprConfig,
    DprModel,
    SetEncoderConfig,
    SetEncoderModel,
    SetEncoderTokenizer,
    SpladeConfig,
    SpladeModel,
    T5CrossEncoderConfig,
    T5CrossEncoderModel,
    T5CrossEncoderTokenizer,
)


def _register_internal_models():
    AutoTokenizer.register(BiEncoderConfig, BiEncoderTokenizer)
    AutoConfig.register(CrossEncoderConfig.model_type, CrossEncoderConfig)
    AutoModel.register(CrossEncoderConfig, CrossEncoderModel)
    AutoTokenizer.register(CrossEncoderConfig, CrossEncoderTokenizer)
    AutoConfig.register(ColConfig.model_type, ColConfig)
    AutoModel.register(ColConfig, ColModel)
    AutoTokenizer.register(ColConfig, ColTokenizer)
    AutoConfig.register(DprConfig.model_type, DprConfig)
    AutoModel.register(DprConfig, DprModel)
    AutoTokenizer.register(DprConfig, BiEncoderTokenizer)
    AutoConfig.register(SetEncoderConfig.model_type, SetEncoderConfig)
    AutoModel.register(SetEncoderConfig, SetEncoderModel)
    AutoTokenizer.register(SetEncoderConfig, SetEncoderTokenizer)
    AutoConfig.register(SpladeConfig.model_type, SpladeConfig)
    AutoModel.register(SpladeConfig, SpladeModel)
    AutoTokenizer.register(SpladeConfig, BiEncoderTokenizer)
    AutoConfig.register(T5CrossEncoderConfig.model_type, T5CrossEncoderConfig)
    AutoModel.register(T5CrossEncoderConfig, T5CrossEncoderModel)
    AutoTokenizer.register(T5CrossEncoderConfig, T5CrossEncoderTokenizer)
