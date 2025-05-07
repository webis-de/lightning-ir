from typing import Dict, Literal, Sequence, Type

import torch
from transformers import BatchEncoding

from ..cross_encoder.cross_encoder_config import CrossEncoderConfig
from ..cross_encoder.cross_encoder_model import CrossEncoderModel, CrossEncoderOutput
from ..cross_encoder.cross_encoder_tokenizer import CrossEncoderTokenizer


class T5CrossEncoderConfig(CrossEncoderConfig):

    model_type = "encoder-decoder-cross-encoder"

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        decoder_strategy: Literal["mono", "rank"] = "mono",
        **kwargs,
    ) -> None:
        kwargs["pooling_strategy"] = "first"
        super().__init__(query_length=query_length, doc_length=doc_length, **kwargs)
        self.decoder_strategy = decoder_strategy


class ScaleLinear(torch.nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586 # noqa
        input = input * (input.shape[-1] ** -0.5)
        return super().forward(input)


class T5CrossEncoderModel(CrossEncoderModel):
    config_class = T5CrossEncoderConfig

    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "linear.weight"]

    def __init__(self, config: T5CrossEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: T5CrossEncoderConfig
        if self.config.decoder_strategy == "mono":
            self.linear = ScaleLinear(config.hidden_size, 2, bias=config.linear_bias)
        else:
            self.linear = ScaleLinear(config.hidden_size, 1, bias=config.linear_bias)

    # TODO tieing of weights does not work when setting linear to only use slice of lm head for efficiency
    # def get_output_embeddings(self):
    #     shared = self.shared
    #     if self.config.decoder_strategy == "mono":
    #         self.linear.weight.data = shared.weight.data[[1176, 6136]]
    #     elif self.config.decoder_strategy == "rank":
    #         self.linear.weight.data = shared.weight.data[[32089]]
    #     else:
    #         raise ValueError("Unknown decoder strategy")
    #     return shared

    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        decoder_input_ids = torch.zeros(
            (encoding["input_ids"].shape[0], 1), device=encoding["input_ids"].device, dtype=torch.long
        )
        encoding["decoder_input_ids"] = decoder_input_ids
        output = super().forward(encoding)
        if output.scores is None:
            raise ValueError("Scores are None")
        if self.config.decoder_strategy == "mono":
            scores = output.scores.view(-1, 2)
            scores = torch.nn.functional.log_softmax(scores, dim=-1)[:, 0]
            output.scores = scores.view(-1)
        return output


class T5CrossEncoderTokenizer(CrossEncoderTokenizer):

    config_class: Type[T5CrossEncoderConfig] = T5CrossEncoderConfig

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        decoder_strategy: Literal["mono", "rank"] = "mono",
        **kwargs,
    ):
        super().__init__(
            *args, query_length=query_length, doc_length=doc_length, decoder_strategy=decoder_strategy, **kwargs
        )
        self.decoder_strategy = decoder_strategy

    def tokenize(
        self,
        queries: str | Sequence[str] | None = None,
        docs: str | Sequence[str] | None = None,
        num_docs: Sequence[int] | int | None = None,
        **kwargs,
    ) -> Dict[str, BatchEncoding]:
        expanded_queries, docs = self._preprocess(queries, docs, num_docs)
        if self.decoder_strategy == "mono":
            pattern = "Query: {query} Document: {doc} Relevant:"
        elif self.decoder_strategy == "rank":
            pattern = "Query: {query} Document: {doc}"
        else:
            raise ValueError(f"Unknown decoder strategy: {self.decoder_strategy}")
        input_texts = [pattern.format(query=query, doc=doc) for query, doc in zip(expanded_queries, docs)]

        return_tensors = kwargs.get("return_tensors", None)
        if return_tensors is not None:
            kwargs["pad_to_multiple_of"] = 8
        return {"encoding": self(input_texts, **kwargs)}
