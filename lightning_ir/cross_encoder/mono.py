from typing import Sequence

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    ElectraConfig,
    ElectraModel,
    ElectraPreTrainedModel,
    RobertaConfig,
    RobertaModel,
    RobertaPreTrainedModel,
)

from ..flash.flash_model import FlashClassFactory
from ..loss.loss import LossFunction
from .model import CrossEncoderConfig, CrossEncoderModel
from .module import CrossEncoderModule


class Pooler(torch.nn.Module):
    def __init__(self, encoder: BertModel | ElectraModel | RobertaModel) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state[:, 0, :]


class MonoBertConfig(BertConfig, CrossEncoderConfig):
    model_type = "mono-bert"


class MonoElectraConfig(ElectraConfig, CrossEncoderConfig):
    model_type = "mono-electra"


class MonoRobertaConfig(RobertaConfig, CrossEncoderConfig):
    model_type = "mono-roberta"


class MonoBertModel(BertPreTrainedModel, CrossEncoderModel):
    config_class = MonoBertConfig

    def __init__(self, mono_bert_config: MonoBertConfig) -> None:
        super().__init__(mono_bert_config, "bert")
        self.bert = BertModel(mono_bert_config, add_pooling_layer=False)


class MonoElectraModel(ElectraPreTrainedModel, CrossEncoderModel):
    config_class = MonoElectraConfig

    def __init__(self, mono_electra_config: MonoElectraConfig) -> None:
        super().__init__(mono_electra_config, "electra")
        self.electra = ElectraModel(mono_electra_config)


class MonoRobertaModel(RobertaPreTrainedModel, CrossEncoderModel):
    config_class = MonoRobertaConfig

    def __init__(self, mono_roberta_config: MonoRobertaConfig) -> None:
        super().__init__(mono_roberta_config, "roberta")
        self.roberta = RobertaModel(mono_roberta_config)


FlashMonoBertModel = FlashClassFactory(MonoBertModel)
FlashMonoElectraModel = FlashClassFactory(MonoElectraModel)
FlashMonoRobertaModel = FlashClassFactory(MonoRobertaModel)


class MonoBertModule(CrossEncoderModule):
    config_class = MonoBertConfig

    def __init__(
        self,
        model: MonoBertModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | MonoBertConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, MonoBertConfig):
                    raise ValueError("To initialize a new model pass a MonoBertConfig.")
                model = FlashMonoBertModel(config)
            else:
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
                model = FlashMonoBertModel.from_pretrained(
                    model_name_or_path, config=config
                )
        super().__init__(model, loss_functions, evaluation_metrics)


class MonoElectraModule(CrossEncoderModule):
    config_class = MonoElectraConfig

    def __init__(
        self,
        model: MonoElectraModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | MonoElectraConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, MonoElectraConfig):
                    raise ValueError(
                        "To initialize a new model pass a MonoElectraConfig."
                    )
                model = FlashMonoElectraModel(config)
            else:
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
                model = FlashMonoElectraModel.from_pretrained(
                    model_name_or_path, config=config
                )
        super().__init__(model, loss_functions, evaluation_metrics)


class MonoRobertaModule(CrossEncoderModule):
    config_class = MonoRobertaConfig

    def __init__(
        self,
        model: MonoRobertaModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | MonoRobertaConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, MonoRobertaConfig):
                    raise ValueError(
                        "To initialize a new model pass a MonoRobertaConfig."
                    )
                model = FlashMonoRobertaModel(config)
            else:
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
                model = FlashMonoRobertaModel.from_pretrained(
                    model_name_or_path, config=config
                )
        super().__init__(model, loss_functions, evaluation_metrics)


AutoConfig.register(MonoBertConfig.model_type, MonoBertConfig)
AutoModel.register(MonoBertConfig, MonoBertModel)
AutoConfig.register(MonoElectraConfig.model_type, MonoElectraConfig)
AutoModel.register(MonoElectraConfig, MonoElectraModel)
AutoConfig.register(MonoRobertaConfig.model_type, MonoRobertaConfig)
AutoModel.register(MonoRobertaConfig, MonoRobertaModel)
