"""Configuration for the vendored NeoBERT backbone.

Origin: ``chandar-lab/NeoBERT`` (Hugging Face Hub), snapshot revision
``5424c8efeea6491b151d62dee55a752165407430`` — ``model.py``/``config.json``.
Original code and weights are MIT licensed (Copyright (c) 2025 Chandar Research Lab).

Modifications relative to the original ``NeoBERTConfig``:
- Defaults corrected to the *published* ``config.json`` values (``norm_eps=1e-5``,
  ``max_length=4096``) so ``NeoBERTConfig()`` reproduces the released architecture.
- ``classifier_init_range`` promoted to an explicit field (it is a top-level field of the
  published ``config.json``); ``tie_word_embeddings`` defaults to ``False`` (the MLM decoder
  is *not* tied to the input embedding — verified against the checkpoint).
- Dropped the original's ``self.kwargs = kwargs`` quirk (it leaked a ``"kwargs"`` blob into
  the serialized config) and the ``auto_map``/``trust_remote_code`` remote-code plumbing.
- Added a ``max_position_embeddings`` alias property over ``max_length``.
"""

from transformers import PretrainedConfig


class NeoBERTConfig(PretrainedConfig):
    """Configuration for the NeoBERT encoder backbone.

    Field names and default values mirror the published ``chandar-lab/NeoBERT`` ``config.json``
    so that ``NeoBERTConfig()`` alone reconstructs the released model.
    """

    model_type = "neobert"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        classifier_init_range: float = 0.02,
        norm_eps: float = 1e-5,
        max_length: int = 4096,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.classifier_init_range = classifier_init_range
        self.norm_eps = norm_eps
        self.max_length = max_length

    @property
    def max_position_embeddings(self) -> int:
        """Alias for :attr:`max_length` — the standard name some transformers utilities read."""
        return self.max_length
