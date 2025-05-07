"""Learning rate schedulers for LightningIR."""

import torch

from .schedulers import ConstantSchedulerWithLinearWarmup, LambdaWarmupScheduler, LinearSchedulerWithLinearWarmup


class WarmupLRScheduler(LambdaWarmupScheduler, torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        **kwargs,
    ) -> None:
        """Base class for learning rate schedulers with warmup.

        :param optimizer: Optimizer to adjust the learning rate for.
        :type optimizer: torch.optim.Optimizer
        :param num_warmup_steps: Number of warmup steps.
        :type num_warmup_steps: int
        """
        last_epoch = -1
        self.interval = "step"
        super().__init__(
            optimizer=optimizer,
            lr_lambda=self.value_lambda,
            num_warmup_steps=num_warmup_steps,
            last_epoch=last_epoch,
            **kwargs,
        )


class LinearLRSchedulerWithLinearWarmup(WarmupLRScheduler, LinearSchedulerWithLinearWarmup):
    """Scheduler for linearly decreasing learning rate with linear warmup."""

    pass


class ConstantLRSchedulerWithLinearWarmup(WarmupLRScheduler, ConstantSchedulerWithLinearWarmup):
    """Scheduler for constant learning rate with linear warmup."""

    pass


LR_SCHEDULERS = (
    LinearLRSchedulerWithLinearWarmup,
    ConstantLRSchedulerWithLinearWarmup,
    WarmupLRScheduler,
    torch.optim.lr_scheduler.LRScheduler,
)
