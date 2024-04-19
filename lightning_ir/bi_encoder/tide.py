import math
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Sequence

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
)

from ..flash.flash_model import FlashClassFactory
from ..loss.loss import LossFunction
from .model import BiEncoderConfig, BiEncoderModel, ScoringFunction
from .module import BiEncoderModule


class TideConfig(BertConfig, BiEncoderConfig):
    model_type = "tide"

    ADDED_ARGS = BiEncoderConfig.ADDED_ARGS + [
        "query_embedding_length",
        "doc_embedding_length",
    ]

    def __init__(
        self, query_embedding_length: int = 8, doc_embedding_length: int = 32, **kwargs
    ) -> None:
        kwargs["query_expansion"] = True
        kwargs["doc_expansion"] = True
        kwargs["attend_to_query_expanded_tokens"] = True
        kwargs["attend_to_doc_expanded_tokens"] = True
        super().__init__(**kwargs)
        self.query_embedding_length = query_embedding_length
        self.doc_embedding_length = doc_embedding_length

    def to_added_args_dict(self) -> Dict[str, Any]:
        mvr_dict = super().to_added_args_dict()
        mvr_dict["query_embedding_length"] = self.query_embedding_length
        mvr_dict["doc_embedding_length"] = self.doc_embedding_length
        return mvr_dict


class TideScoringFunction(ScoringFunction):
    def query_scoring_mask(
        self,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_input_ids is not None:
            query_input_ids = query_input_ids[:, : self.config.query_embedding_length]
        if query_attention_mask is not None:
            query_attention_mask = query_attention_mask[
                :, : self.config.query_embedding_length
            ]
        return super().query_scoring_mask(query_input_ids, query_attention_mask)

    def doc_scoring_mask(
        self,
        doc_input_ids: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if doc_input_ids is not None:
            doc_input_ids = doc_input_ids[:, : self.config.doc_embedding_length]
        if doc_attention_mask is not None:
            doc_attention_mask = doc_attention_mask[
                :, : self.config.doc_embedding_length
            ]
        return super().doc_scoring_mask(doc_input_ids, doc_attention_mask)


class TideModel(BertPreTrainedModel, BiEncoderModel):
    config_class = TideConfig

    def __init__(self, tide_config: TideConfig):
        bert = BertModel(tide_config, add_pooling_layer=False)
        super().__init__(tide_config, bert)
        self._modules["bert"] = self._modules.pop("encoder")
        query_embedding_lengths = self.get_embedding_lengths(
            self.config.query_length, self.config.query_embedding_length
        )
        doc_embedding_lengths = self.get_embedding_lengths(
            self.config.doc_length, self.config.doc_embedding_length
        )
        self.pooling_context_manager = PoolingContextManager(
            self, query_embedding_lengths, doc_embedding_lengths
        )
        self.scoring_function = TideScoringFunction(tide_config)

    def get_embedding_lengths(
        self, sequence_length: int, embedding_length: int
    ) -> List[int]:
        # reduction = (embedding_length / sequence_length) ** (
        #     -1 / self.config.num_hidden_layers
        # )
        embedding_lengths = []
        for _ in range(self.config.num_hidden_layers):
            embedding_lengths.insert(0, embedding_length)
            embedding_length = min(sequence_length, embedding_length * 2)
            # embedding_lengths.append(
            #     max(
            #         int(sequence_length / reduction ** (layer_idx + 1)),
            #         embedding_length,
            #     )
            # )
        if embedding_lengths[0] != sequence_length:
            raise ValueError("Unable to match embedding length to sequence length.")
        # if embedding_lengths[-1] != embedding_length:
        #     raise ValueError(
        #         "The final embedding length is unequal to the desired embedding length."
        #     )
        return embedding_lengths

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self.pooling_context_manager("query"):
            embedding = super().encode_queries(
                input_ids, attention_mask, token_type_ids
            )
        return embedding

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self.pooling_context_manager("doc"):
            embedding = super().encode_docs(input_ids, attention_mask, token_type_ids)
        return embedding

    @property
    def encoder(self):
        return self.bert


class PoolingContextManager:
    def __init__(
        self,
        model: TideModel,
        query_embedding_lengths: List[int],
        doc_embedding_lengths: List[int],
    ) -> None:
        self.model = model
        self.query_embedding_lengths = query_embedding_lengths
        self.doc_embedding_lengths = doc_embedding_lengths
        self.query_layers = []
        self.query_forwards = []
        self.output_layers = []
        self.output_forwards = []
        for name, module in self.model.named_modules():
            if name.endswith("query"):
                self.query_layers.append(module)
                self.query_forwards.append(module.forward)
            if name.endswith("attention.output"):
                self.output_layers.append(module)
                self.output_forwards.append(module.forward)
        if len(self.query_layers) != len(self.output_layers):
            raise ValueError("The number of query and output layers must be equal.")
        self.input_type: Literal["query", "doc"] | None = None

    @staticmethod
    def pool_output_hidden_states(
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        output_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        embedding_length: int,
    ) -> torch.Tensor:
        if embedding_length != input_tensor.shape[-2]:
            input_tensor = torch.nn.functional.adaptive_avg_pool1d(
                input_tensor.transpose(-1, -2), embedding_length
            ).transpose(-1, -2)
        return output_forward(hidden_states, input_tensor)

    @staticmethod
    def pool_query_hidden_states(
        hidden_states: torch.Tensor,
        linear_forward: Callable[[torch.Tensor], torch.Tensor],
        embedding_length: int,
    ) -> torch.Tensor:
        hidden_states = linear_forward(hidden_states)
        if embedding_length != hidden_states.shape[-2]:
            hidden_states = torch.nn.functional.adaptive_avg_pool1d(
                hidden_states.transpose(-1, -2), embedding_length
            ).transpose(-1, -2)
        return hidden_states

    def __call__(self, input_type: Literal["query", "doc"]) -> "PoolingContextManager":
        self.input_type = input_type
        return self

    def __enter__(self) -> "PoolingContextManager":
        if self.input_type == "query":
            embedding_lengths = self.query_embedding_lengths
        elif self.input_type == "doc":
            embedding_lengths = self.doc_embedding_lengths
        else:
            raise ValueError("input_type must be 'query' or 'doc'")
        iterator = zip(self.query_layers, self.output_layers, embedding_lengths)
        for query_layer, output_layer, embedding_length in iterator:
            query_layer.forward = partial(
                self.pool_query_hidden_states,
                linear_forward=query_layer.forward,
                embedding_length=embedding_length,
            )
            output_layer.forward = partial(
                self.pool_output_hidden_states,
                output_forward=output_layer.forward,
                embedding_length=embedding_length,
            )
        return self

    def __exit__(self, *args, **kwargs):
        self.input_type = None
        iterator = zip(
            self.query_layers,
            self.query_forwards,
            self.output_layers,
            self.output_forwards,
        )
        for query_layer, query_forward, output_layer, output_forward in iterator:
            query_layer.forward = query_forward
            output_layer.forward = output_forward


FlashTideModel = FlashClassFactory(TideModel)


class TideModule(BiEncoderModule):
    def _bi_encoder(
        self,
        model_name_or_path: str | None = None,
        config: BiEncoderConfig | TideConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ) -> None:
        if model_name_or_path is None:
            if config is None:
                raise ValueError(
                    "Either model_name_or_path or config must be provided."
                )
            if not isinstance(config, TideConfig):
                raise ValueError("config initializing a new model pass a TideConfig.")
            model = FlashTideModel(config)
        else:
            model = FlashTideModel.from_pretrained(model_name_or_path, config=config)
        super().__init__(model, loss_functions, evaluation_metrics)


AutoConfig.register("tide", TideConfig)
AutoModel.register(TideConfig, TideModel)
