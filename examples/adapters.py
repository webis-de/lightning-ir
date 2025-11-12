#!/usr/bin/env python3
"""
Example demonstrating LoRA adapter usage with Lightning IR models.

Note: This example requires the 'adapters' extra dependency:
pip install lightning-ir[adapters]
"""

from peft import LoraConfig
from torch.optim import AdamW

from lightning_ir import BiEncoderModule, LightningIRDataModule, LightningIRTrainer, RankNet, TupleDataset
from lightning_ir.models import DprConfig

# Define the model
module = BiEncoderModule(
    model_name_or_path="bert-base-uncased",  # backbone model
    config=DprConfig(
        use_adapter=True,  # Enable adapter use
        adapter_config=LoraConfig(  # Specify the adapter config
            r=16, lora_alpha=32, target_modules=["query", "key", "value"], lora_dropout=0.1
        ),
    ),
    loss_functions=[RankNet()],
)
module.set_optimizer(AdamW, lr=1e-5)

# Define the data module
data_module = LightningIRDataModule(
    train_dataset=TupleDataset("msmarco-passage/train/triples-small"),
    train_batch_size=32,
)

# Define the trainer
trainer = LightningIRTrainer(max_steps=100_000)

# Fine-tune the model
trainer.fit(module, data_module)
