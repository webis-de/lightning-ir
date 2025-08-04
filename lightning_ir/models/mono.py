"""
Model implementation for mono cross-encoder models. Originally introduced in
`Passage Re-ranking with BERT
<https://arxiv.org/abs/1901.04085>`_.
"""

from typing import Literal, Type

import torch
from transformers import BatchEncoding

from ..base.model import batch_encoding_wrapper
from ..cross_encoder.cross_encoder_config import CrossEncoderConfig
from ..cross_encoder.cross_encoder_model import CrossEncoderModel, CrossEncoderOutput


class ScaleLinear(torch.nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586 # noqa
        input = input * (input.shape[-1] ** -0.5)
        return super().forward(input)


class MonoConfig(CrossEncoderConfig):
    """Configuration class for mono cross-encoder models."""

    model_type = "mono"
    """Model type for mono cross-encoder models."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        pooling_strategy: Literal["first", "mean", "max", "sum", "bert_pool"] = "first",
        linear_bias: bool = False,
        scoring_strategy: Literal["mono", "rank"] = "rank",
        tokenizer_pattern: str | None = None,
        **kwargs,
    ):
        """Initialize the configuration for mono cross-encoder models.

        Args:
            query_length (int): Maximum query length. Defaults to 32.
            doc_length (int): Maximum document length. Defaults to 512.
            pooling_strategy (Literal["first", "mean", "max", "sum", "bert_pool"]): Pooling strategy for the
                embeddings. Defaults to "first".
            linear_bias (bool): Whether to use bias in the final linear layer. Defaults to False.
            scoring_strategy (Literal["mono", "rank"]): Scoring strategy to use. Defaults to "rank".
            tokenizer_pattern (str | None): Optional pattern for tokenization. Defaults to None.
        """
        self._bert_pool = False
        if pooling_strategy == "bert_pool":
            self._bert_pool = True
            pooling_strategy = "first"
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            pooling_strategy=pooling_strategy,
            linear_bias=linear_bias,
            **kwargs,
        )
        self.scoring_strategy = scoring_strategy
        self.tokenizer_pattern = tokenizer_pattern


class MonoModel(CrossEncoderModel):
    config_class: Type[MonoConfig] = MonoConfig
    """Configuration class for mono cross-encoder models."""

    def __init__(self, config: MonoConfig, *args, **kwargs):
        """A cross-encoder model that jointly encodes a query and document(s). The contextualized embeddings are
        aggragated into a single vector and fed to a linear layer which computes a final relevance score.

        Args:
            config (MonoConfig): Configuration for the mono cross-encoder model.
        """
        super().__init__(config, *args, **kwargs)

        if self.config.scoring_strategy == "mono":
            output_dim = 2
        elif self.config.scoring_strategy == "rank":
            output_dim = 1
        else:
            raise ValueError(
                f"Unknown scoring strategy {self.config.scoring_strategy}. Supported strategies are 'mono' and 'rank'."
            )

        self.bert_pool = torch.nn.Identity()
        if self.config._bert_pool:
            self.bert_pool = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size), torch.nn.Tanh()
            )

        if self.config.backbone_model_type == "t5":
            self.linear = ScaleLinear(config.hidden_size, output_dim, bias=self.config.linear_bias)
        else:
            self.linear = torch.nn.Linear(config.hidden_size, output_dim, bias=self.config.linear_bias)

    @batch_encoding_wrapper
    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        """Computes contextualized embeddings for the joint query-document input sequence and computes a relevance
        score.

        Args:
            encoding (BatchEncoding): Tokenizer encodings for the joint query-document input sequence.
        Returns:
            CrossEncoderOutput: Output of the model.
        """
        if hasattr(self, "decoder"):
            # NOTE hack to make T5 cross-encoders work. other encoder-decoder models may not have `decoder` as their
            # attribute. maybe find a better way to check for this?
            decoder_input_ids = torch.zeros(
                (encoding["input_ids"].shape[0], 1), device=encoding["input_ids"].device, dtype=torch.long
            )
            encoding["decoder_input_ids"] = decoder_input_ids
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.pooling(
            embeddings, encoding.get("attention_mask", None), pooling_strategy=self.config.pooling_strategy
        )
        embeddings = self.bert_pool(embeddings)
        scores = self.linear(embeddings)

        if self.config.scoring_strategy == "mono":
            scores = torch.nn.functional.log_softmax(scores.view(-1, 2), dim=-1)[:, 1]

        return CrossEncoderOutput(scores=scores.view(-1), embeddings=embeddings)
