from abc import ABC, abstractmethod

import torch


class WarmupLRScheduler(torch.optim.lr_scheduler.LambdaLR, ABC):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        verbose: bool = False,
    ) -> None:
        last_epoch = -1
        self.interval = "step"
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch, verbose)

    @abstractmethod
    def lr_lambda(self, current_step: int) -> float:
        ...


class LinearLRSchedulerWithLinearWarmup(WarmupLRScheduler):
    def lr_lambda(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return current_step / self.num_warmup_steps
        return max(
            0.0,
            (self.num_training_steps - current_step) / (self.num_training_steps - self.num_warmup_steps),
        )


class ConstantLRSchedulerWithLinearWarmup(WarmupLRScheduler):
    def lr_lambda(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return current_step / self.num_warmup_steps
        return 1.0


LR_SCHEDULERS = [LinearLRSchedulerWithLinearWarmup, ConstantLRSchedulerWithLinearWarmup]
