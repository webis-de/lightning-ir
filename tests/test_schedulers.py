from typing import Any

import pytest
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from lightning_ir.lightning_utils.schedulers import (
    ConstantSchedulerWithWarmup,
    LambdaWarmupScheduler,
    LinearSchedulerWithWarmup,
)


class DummyObject:
    def __init__(self) -> None:
        self.value = 100


class DummyModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))
        self.dummy_object = DummyObject()

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DummyDataset(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, index) -> Any:
        return 0


class DummyDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def train_dataloader(self) -> Any:
        return DataLoader(DummyDataset(), batch_size=1)


@pytest.mark.parametrize(
    "scheduler",
    (
        LinearSchedulerWithWarmup(["dummy_object.value"], 20, 100),
        ConstantSchedulerWithWarmup(["dummy_object.value"], 20, 100),
    ),
    ids=["Linear", "Constant"],
)
def test_scheduler(scheduler: LambdaWarmupScheduler):
    trainer = Trainer(
        callbacks=[scheduler],
        max_steps=10,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    module = DummyModule()
    trainer.fit(module, datamodule=DummyDataModule())

    assert scheduler.values["dummy_object.value"] == 100
    assert module.dummy_object.value == 50
