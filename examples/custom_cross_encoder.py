import torch
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, AutoTokenizer, BatchEncoding

from lightning_ir import (
    CrossEncoderModel,
    CrossEncoderModule,
    CrossEncoderOutput,
    CrossEncoderTokenizer,
    LightningIRDataModule,
    LightningIRTrainer,
    RankNet,
    TupleDataset,
)
from lightning_ir.cross_encoder.config import CrossEncoderConfig


class CustomCrossEncoderConfig(CrossEncoderConfig):
    model_type = "custom-cross-encoder"

    ADDED_ARGS = CrossEncoderConfig.ADDED_ARGS.union({"additional_linear_layer"})

    def __init__(self, additional_linear_layer: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.additional_linear_layer = additional_linear_layer


class CustomCrossEncoderModel(CrossEncoderModel):
    config_class = CustomCrossEncoderConfig

    def __init__(self, config: CustomCrossEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.additional_linear_layer = None
        if config.additional_linear_layer:
            self.additional_linear_layer = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, encoding: BatchEncoding) -> torch.Tensor:
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self._pooling(
            embeddings, encoding.get("attention_mask", None), pooling_strategy=self.config.pooling_strategy
        )
        if self.additional_linear_layer is not None:
            embeddings = self.additional_linear_layer(embeddings)
        scores = self.linear(embeddings).view(-1)
        return CrossEncoderOutput(scores=scores, embeddings=embeddings)


# register the config, model and tokenizer with the transformers Auto* classes
AutoConfig.register(CustomCrossEncoderConfig.model_type, CustomCrossEncoderConfig)
AutoModel.register(CustomCrossEncoderConfig, CustomCrossEncoderModel)
AutoTokenizer.register(CustomCrossEncoderConfig, CrossEncoderTokenizer)

# Fine-tune our custom model
module = CrossEncoderModule(
    model_name_or_path="bert-base-uncased",
    config=CustomCrossEncoderConfig(),  # our custom config
    loss_functions=[RankNet()],
)
module.set_optimizer(AdamW, lr=1e-5)
data_module = LightningIRDataModule(
    train_dataset=TupleDataset("msmarco-passage/train/triples-small"), train_batch_size=32
)
trainer = LightningIRTrainer(max_steps=100_000)
trainer.fit(module, data_module)
