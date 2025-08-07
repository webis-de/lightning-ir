from lightning_ir import (
    BiEncoderModule,
    LightningIRDataModule,
    LightningIRTrainer,
    QueryDataset,
    SearchCallback,
    TorchDenseSearchConfig,
)

# Define the model
module = BiEncoderModule(
    model_name_or_path="webis/bert-bi-encoder",
    evaluation_metrics=["nDCG@10"],
)

# Define the data module
data_module = LightningIRDataModule(
    inference_datasets=[
        QueryDataset("msmarco-passage/trec-dl-2019/judged"),
        QueryDataset("msmarco-passage/trec-dl-2020/judged"),
    ],
    inference_batch_size=4,
)

# Define the search callback
callback = SearchCallback(
    index_dir="./msmarco-passage-index",
    search_config=TorchDenseSearchConfig(k=100),
    save_dir="./runs",
)

# Define the trainer
trainer = LightningIRTrainer(callbacks=[callback], logger=False, enable_checkpointing=False)

# Retrieve relevant documents
trainer.search(module, data_module)
