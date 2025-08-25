from torch.optim import AdamW

from lightning_ir import BiEncoderModule, LightningIRDataModule, LightningIRTrainer, RankNet, TupleDataset
from lightning_ir.models import DprConfig

# Define the model
module = BiEncoderModule(
    model_name_or_path="bert-base-uncased",  # backbone model
    config=DprConfig(),
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
