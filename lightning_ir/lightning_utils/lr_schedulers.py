import torch

from .schedulers import ConstantSchedulerWithLinearWarmup, LambdaWarmupScheduler, LinearSchedulerWithLinearWarmup


class WarmupLRScheduler(LambdaWarmupScheduler, torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        *args,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        last_epoch = -1
        self.interval = "step"
        super().__init__(
            *args,
            optimizer=optimizer,
            lr_lambda=self.value_lambda,
            num_warmup_steps=num_warmup_steps,
            last_epoch=last_epoch,
            verbose=verbose,
            **kwargs,
        )


class LinearLRSchedulerWithLinearWarmup(WarmupLRScheduler, LinearSchedulerWithLinearWarmup):
    pass


class ConstantLRSchedulerWithLinearWarmup(WarmupLRScheduler, ConstantSchedulerWithLinearWarmup):
    pass


LR_SCHEDULERS = [LinearLRSchedulerWithLinearWarmup, ConstantLRSchedulerWithLinearWarmup]
