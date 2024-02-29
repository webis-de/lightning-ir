from typing import Dict, Any, Sequence
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
)

from mvr.flash.flash_model import FlashClassFactory
from mvr.loss import LossFunction
from mvr.mvr import MVRConfig, MVRModel, MVRModule


class TideConfig(BertConfig, MVRConfig):
    model_type = "tide"

    def __init__(
        self, query_embedding_length: int, doc_embedding_length: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.query_embedding_length = query_embedding_length
        self.doc_embedding_length = doc_embedding_length

    def to_mvr_dict(self) -> Dict[str, Any]:
        mvr_dict = super().to_mvr_dict()
        mvr_dict["num_embeddings"] = self.num_embeddings
        return mvr_dict


class TideQueryLayer(torch.nn.Module):

    def __init__(self, layer_idx: int, config: TideConfig) -> None:
        super().__init__()
        num_attention_heads = config.num_attention_heads
        attention_head_size = int(config.hidden_size / config.num_attention_heads)
        all_head_size = num_attention_heads * attention_head_size

        self.pooling = torch.nn.AdaptiveAvgPool1d(config.num_embeddings[layer_idx])
        self.linear = torch.nn.Linear(config.hidden_size, all_head_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pooling(hidden_states)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class TideModel(BertPreTrainedModel, MVRModel):
    config_class = TideConfig

    def __init__(self, tide_config: TideConfig):
        bert = BertModel(tide_config, add_pooling_layer=False)
        super().__init__(tide_config, bert)
        self._modules["bert"] = self._modules.pop("encoder")
        for name, module in self.named_modules():
            if name.endswith("query"):
                foo

    @property
    def encoder(self):
        return self.bert


FlashTideModel = FlashClassFactory(TideModel)


class TideModule(MVRModule):

    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: MVRConfig | TideConfig | None = None,
        loss_function: LossFunction | None = None,
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
        super().__init__(model, loss_function)


AutoConfig.register("tide", TideConfig)
AutoModel.register(TideConfig, TideModel)
