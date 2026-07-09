"""Register vendored third-party backbones with the transformers Auto registries.

Mirrors ``register_internal_models.py``. Called from ``lightning_ir/__init__.py`` at import time, so
``import lightning_ir`` makes NeoBERT usable through the (unmodified) mvrt5 class factory exactly like
BERT/ModernBERT — resolving config, model, and tokenizer by type with no ``trust_remote_code``.
"""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
)

from .backbones.neobert import NeoBERTConfig, NeoBERTForMaskedLM, NeoBERTModel


def _register_backbones():
    AutoConfig.register(NeoBERTConfig.model_type, NeoBERTConfig, exist_ok=True)
    AutoModel.register(NeoBERTConfig, NeoBERTModel, exist_ok=True)
    AutoModelForMaskedLM.register(NeoBERTConfig, NeoBERTForMaskedLM, exist_ok=True)
    # NeoBERT ships a BERT-family (BertTokenizer) tokenizer. Registering it is what keeps
    # base/tokenizer.py untouched: the factory looks tokenizers up by backbone config class (F5).
    AutoTokenizer.register(
        NeoBERTConfig,
        slow_tokenizer_class=BertTokenizer,
        fast_tokenizer_class=BertTokenizerFast,
        exist_ok=True,
    )
