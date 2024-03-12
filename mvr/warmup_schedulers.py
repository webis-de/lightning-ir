import torch


class LinearSchedulerWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        verbose: bool = False,
    ) -> None:
        last_epoch = -1
        self.interval = "step"

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


class ConstantSchedulerWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        verbose: bool = False,
    ) -> None:
        last_epoch = -1
        self.interval = "step"

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


LR_SCHEDULERS = [LinearSchedulerWithWarmup, ConstantSchedulerWithWarmup]
