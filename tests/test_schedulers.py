from typing import Any, Dict, Type

import pytest
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from lightning_ir.schedulers.lr_schedulers import (
    ConstantLRSchedulerWithLinearWarmup,
    LinearLRSchedulerWithLinearWarmup,
    WarmupLRScheduler,
)
from lightning_ir.schedulers.schedulers import (
    GenericConstantSchedulerWithLinearWarmup,
    GenericConstantSchedulerWithQuadraticWarmup,
    GenericLinearSchedulerWithLinearWarmup,
    GenericScheduler,
)


class DummyObject:
    def __init__(self) -> None:
        self.value = 100


class DummyModule(LightningModule):
    def __init__(
        self, LRScheduler: Type[WarmupLRScheduler] | None = None, scheduler_kwargs: Dict[str, float] | None = None
    ) -> None:
        super().__init__()
        self.LRScheduler = LRScheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.param = torch.nn.Parameter(torch.tensor(0.0))
        self.dummy_object = DummyObject()

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self):
        lr = 1
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        out = {"optimizer": optimizer}
        if self.LRScheduler is not None:
            out["lr_scheduler"] = {
                "scheduler": self.LRScheduler(optimizer, **self.scheduler_kwargs),
                "interval": "step",
            }
        return (out,)


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
    "Scheduler,kwargs,intermediate_value,final_value",
    (
        (
            GenericLinearSchedulerWithLinearWarmup,
            {"num_delay_steps": 10, "num_warmup_steps": 20, "num_training_steps": 100},
            50,
            0,
        ),
        (
            GenericConstantSchedulerWithLinearWarmup,
            {"num_delay_steps": 10, "num_warmup_steps": 20},
            50,
            100,
        ),
        (
            GenericConstantSchedulerWithQuadraticWarmup,
            {"num_delay_steps": 10, "num_warmup_steps": 20},
            25,
            100,
        ),
        (
            LinearLRSchedulerWithLinearWarmup,
            {"num_delay_steps": 10, "num_warmup_steps": 20, "num_training_steps": 100},
            0.5,
            0.0,
        ),
        (
            LinearLRSchedulerWithLinearWarmup,
            {"num_delay_steps": 10, "num_warmup_steps": 20, "num_training_steps": 100, "final_value": 0.2},
            0.5,
            0.2,
        ),
        (
            ConstantLRSchedulerWithLinearWarmup,
            {"num_delay_steps": 10, "num_warmup_steps": 20},
            0.5,
            1.0,
        ),
    ),
    ids=["Linear", "Constant", "Quadratic", "LinearLR", "LinearLRFinal", "ConstantLR"],
)
def test_scheduler(
    Scheduler: Type[GenericScheduler | WarmupLRScheduler],
    kwargs: Dict[str, float],
    intermediate_value: float,
    final_value: float,
):
    if issubclass(Scheduler, WarmupLRScheduler):
        module = DummyModule(Scheduler, kwargs)
        callbacks = []
    elif issubclass(Scheduler, GenericScheduler):
        module = DummyModule()
        callbacks = [Scheduler(keys=["dummy_object.value"], **kwargs)]
        callbacks[0].on_train_end = lambda x, y: None
    else:
        raise ValueError("Invalid scheduler")

    trainer = Trainer(
        callbacks=callbacks,
        max_steps=20,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=DummyDataModule())

    if issubclass(Scheduler, WarmupLRScheduler):
        optimizer = module.optimizers()
        assert optimizer.param_groups[0]["lr"] == intermediate_value
    elif issubclass(Scheduler, GenericScheduler):
        assert callbacks[0].values["dummy_object.value"] == 100
        assert module.dummy_object.value == intermediate_value

    if issubclass(Scheduler, WarmupLRScheduler):
        module = DummyModule(Scheduler, kwargs)
        callbacks = []
    elif issubclass(Scheduler, GenericScheduler):
        module = DummyModule()
        callbacks = [Scheduler(keys=["dummy_object.value"], **kwargs)]
        callbacks[0].on_train_end = lambda x, y: None
    else:
        raise ValueError("Invalid scheduler")

    trainer = Trainer(
        callbacks=callbacks,
        max_steps=100,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=DummyDataModule())

    if issubclass(Scheduler, WarmupLRScheduler):
        optimizer = module.optimizers()
        assert optimizer.param_groups[0]["lr"] == final_value
    elif issubclass(Scheduler, GenericScheduler):
        assert callbacks[0].values["dummy_object.value"] == 100
        assert module.dummy_object.value == final_value
