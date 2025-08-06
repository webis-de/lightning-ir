from lightning_ir import (
    BiEncoderModule,
    DocDataset,
    IndexCallback,
    LightningIRDataModule,
    LightningIRTrainer,
    TorchDenseIndexConfig,
)

# Define the model
module = BiEncoderModule(
    model_name_or_path="webis/bert-bi-encoder",
)

# Define the data module
data_module = LightningIRDataModule(
    inference_datasets=[DocDataset("msmarco-passage")],
    inference_batch_size=256,
)

# Define the index callback
callback = IndexCallback(
    index_dir="./msmarco-passage-index",
    index_config=TorchDenseIndexConfig(),
)

# Define the trainer
trainer = LightningIRTrainer(callbacks=[callback], logger=False, enable_checkpointing=False)

# Index the data
trainer.index(module, data_module)
