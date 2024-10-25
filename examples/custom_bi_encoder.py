from typing import Literal

import torch
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, AutoTokenizer, BatchEncoding

from lightning_ir import (
    BiEncoderConfig,
    BiEncoderEmbedding,
    BiEncoderModel,
    BiEncoderModule,
    BiEncoderTokenizer,
    LightningIRDataModule,
    LightningIRTrainer,
    RankNet,
    TupleDataset,
)


class CustomBiEncoderConfig(BiEncoderConfig):
    model_type = "custom-bi-encoder"

    ADDED_ARGS = BiEncoderConfig.ADDED_ARGS.union({"additional_linear_layer"})

    def __init__(self, additional_linear_layer=True, **kwargs):
        super().__init__(**kwargs)
        self.additional_linear_layer = additional_linear_layer


class CustomBiEncoderModel(BiEncoderModel):
    config_class = CustomBiEncoderConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.additional_linear_layer = None
        if config.additional_linear_layer:
            self.additional_linear_layer = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def _encode(
        self,
        encoding: BatchEncoding,
        expansion: bool = False,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
        mask_scoring_input_ids: torch.Tensor | None = None,
    ) -> BiEncoderEmbedding:
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        if self.additional_linear_layer is not None:  # apply additional linear layer
            embeddings = self.additional_linear_layer(embeddings)
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        embeddings = self._sparsification(embeddings, self.config.sparsification)
        embeddings = self._pooling(embeddings, encoding["attention_mask"], pooling_strategy)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self._scoring_mask(
            encoding["input_ids"],
            encoding["attention_mask"],
            expansion,
            pooling_strategy,
            mask_scoring_input_ids,
        )
        return BiEncoderEmbedding(embeddings, scoring_mask)


AutoConfig.register(CustomBiEncoderConfig.model_type, CustomBiEncoderConfig)
AutoModel.register(CustomBiEncoderConfig, CustomBiEncoderModel)
AutoTokenizer.register(CustomBiEncoderConfig, BiEncoderTokenizer)

module = BiEncoderModule(
    model_name_or_path="bert-base-uncased",
    config=CustomBiEncoderConfig(),  # our custom config
    loss_functions=[RankNet()],
)
module.set_optimizer(AdamW, lr=1e-5)
data_module = LightningIRDataModule(
    train_dataset=TupleDataset("msmarco-passage/train/triples-small"),
    train_batch_size=2,
)
trainer = LightningIRTrainer(max_steps=100_000)
trainer.fit(module, data_module)
