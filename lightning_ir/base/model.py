from dataclasses import dataclass
from typing import Type

import torch
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput

from . import LightningIRConfig


@dataclass
class LightningIROutput(ModelOutput):
    scores: torch.Tensor | None = None


class LightningIRModel(PreTrainedModel):
    config_class = LightningIRConfig

    def __init__(self, config: LightningIRConfig) -> None:
        super().__init__(config)
        self.config = config

    def backbone_forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> LightningIROutput:
        raise NotImplementedError

    def pooling(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.config.pooling_strategy is None:
            return embeddings
        if self.config.pooling_strategy == "cls":
            return embeddings[:, [0]]
        if self.config.pooling_strategy in ("sum", "mean"):
            if attention_mask is not None:
                embeddings = embeddings * attention_mask.unsqueeze(-1)
            embeddings = embeddings.sum(dim=1)
            if self.config.pooling_strategy == "mean":
                if attention_mask is not None:
                    embeddings = embeddings / attention_mask.sum(dim=1, keepdim=True)
            return embeddings
        if self.config.pooling_strategy == "max":
            if attention_mask is not None:
                embeddings = embeddings.masked_fill(
                    ~attention_mask.bool().unsqueeze(-1), -1e9
                )
            return embeddings.max(dim=1).values
        raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "LightningIRModel":
        if not any(not issubclass(base, LightningIRModel) for base in cls.__bases__):
            config = AutoConfig.from_pretrained(*args, **kwargs)
            BackBoneModel = MODEL_MAPPING[CONFIG_MAPPING[config.backbone_model_type]]
            cls = LightningIRModelClassFactory(BackBoneModel, cls.config_class)
        return super(LightningIRModel, cls).from_pretrained(*args, **kwargs)


def LightningIRModelClassFactory(
    BackboneModel: Type[PreTrainedModel],
    MixinConfig: Type[LightningIRConfig] | None = None,
) -> Type[LightningIRModel]:
    if issubclass(BackboneModel, LightningIRModel):
        return BackboneModel

    BackboneConfig = BackboneModel.config_class
    if BackboneConfig is None or not issubclass(BackboneConfig, PretrainedConfig):
        raise ValueError(f"config_class not found in {BackboneModel.__name__}")

    if MixinConfig is None or not issubclass(MixinConfig, LightningIRConfig):
        raise ValueError(
            f"Model {BackboneModel} is not a LightningIRModel, pass a "
            "LightningIRConfig to create one."
        )

    lir_model_type = MixinConfig.model_type
    LightningIRModelMixin: Type[LightningIRModel] | None = MODEL_MAPPING[MixinConfig]
    if LightningIRModelMixin is None:
        raise ValueError(
            f"MixinConfig class {MixinConfig.__name__} does not have a model_mixin, "
            "pass a valid LightningIRConfig to create a LightningIRModel."
        )

    cc_lir_model_type = "".join(s.title() for s in lir_model_type.split("-"))
    cc_model_type = BackboneConfig.__name__[:-6]
    ModelConfig = type(
        f"{cc_model_type}{cc_lir_model_type}Config",
        (MixinConfig, BackboneConfig),
        {
            "backbone_model_type": BackboneConfig.model_type,
        },
    )

    DerivedLightningIRModel = type(
        f"{cc_model_type}{cc_lir_model_type}Model",
        (LightningIRModelMixin, BackboneModel),
        {"config_class": ModelConfig, "backbone_forward": BackboneModel.forward},
    )

    return DerivedLightningIRModel
