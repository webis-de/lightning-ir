from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import MODEL_MAPPING, AutoConfig
from transformers.activations import ACT2FN
from transformers.modeling_utils import load_state_dict

from ...base import LightningIRModel, LightningIRModelClassFactory
from ...bi_encoder import BiEncoderModel
from .config import SpladeConfig


class MLMHead(torch.nn.Module):
    def __init__(self, config: SpladeConfig) -> None:
        super().__init__()
        self.config = config
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.decoder = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SpladeModel(BiEncoderModel):
    config_class = SpladeConfig

    _tied_weights_keys = ["projection.decoder.bias", "projection.decoder.weight"]

    def __init__(self, config: SpladeConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: SpladeConfig

        if self.config.projection == "mlm":
            self.projection = MLMHead(config)

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str | Path, *args, **kwargs
    ) -> "SpladeModel":
        config = AutoConfig.from_pretrained(model_name_or_path)
        mlm = any(
            architecture.endswith("ForMaskedLM")
            for architecture in config.architectures
        )
        if mlm:
            return cls.from_mlm_checkpoint(model_name_or_path)
        return super().from_pretrained(model_name_or_path, *args, **kwargs)

    def get_output_embeddings(self):
        return self.projection.decoder

    @classmethod
    def from_mlm_checkpoint(
        cls, model_name_or_path: str | Path, *args, **kwargs
    ) -> "SpladeModel":
        config = AutoConfig.from_pretrained(model_name_or_path)
        BackboneModel = MODEL_MAPPING[config.__class__]
        cls = LightningIRModelClassFactory(BackboneModel, SpladeConfig)
        model = super(LightningIRModel, cls).from_pretrained(
            model_name_or_path, *args, add_pooling_layer=False, **kwargs
        )
        try:
            state_dict_path = hf_hub_download(
                repo_id=str(model_name_or_path), filename="model.safetensors"
            )
        except Exception:
            state_dict_path = hf_hub_download(
                repo_id=str(model_name_or_path), filename="pytorch_model.bin"
            )
        state_dict = load_state_dict(state_dict_path)
        state_dict.pop("cls.predictions.bias", None)
        for key in list(state_dict.keys()):
            if not key.startswith("cls"):
                state_dict.pop(key)
                continue
            new_key = key.replace("cls.predictions", "projection").replace(
                ".transform", ""
            )
            state_dict[new_key] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)
        return model
