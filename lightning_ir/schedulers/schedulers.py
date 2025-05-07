"""Generic schedulers for LightningIR."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from lightning import Callback, LightningModule, Trainer

from ..base import LightningIRModule

# TODO add final value to all schedulers
# TODO add cosine decay scheduler


class LambdaWarmupScheduler(ABC):
    def __init__(
        self,
        num_warmup_steps: int,
        num_delay_steps: int = 0,
        *args,
        **kwargs,
    ) -> None:
        """Base class for schedulers with warmup.

        :param num_warmup_steps: Number of warmup steps
        :type num_warmup_steps: int
        :param num_delay_steps: Number of steps to delay scheduler for, defaults to 0
        :type num_delay_steps: int, optional
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_delay_steps = num_delay_steps
        super().__init__(*args, **kwargs)

    @abstractmethod
    def value_lambda(self, current_step: int) -> float:
        """Lambda function to adjust the value at each step.

        :param current_step: Current step
        :type current_step: int
        :return: Value at the current step
        :rtype: float
        """
        ...

    def _check_delay(self, current_step: int) -> bool:
        return current_step < self.num_delay_steps

    def _check_warmup(self, current_step: int) -> bool:
        return current_step < self.num_warmup_steps + self.num_delay_steps


class LinearSchedulerWithLinearWarmup(LambdaWarmupScheduler):

    def __init__(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
        *args,
        final_value: float = 0.0,
        num_delay_steps: int = 0,
        **kwargs,
    ) -> None:
        """Scheduler for linearly decreasing values with linear warmup.

        :param num_warmup_steps: Number of warmup steps
        :type num_warmup_steps: int
        :param num_training_steps: Number of training steps
        :type num_training_steps: int
        :param final_value: The final value that should be reached at the end of decay, defaults to 0.0
        :type final_value: float, optional
        :param num_delay_steps: Number of steps to delay warmup / decay, defaults to 0
        :type num_delay_steps: int, optional
        """
        self.num_training_steps = num_training_steps
        self.final_value = final_value
        super().__init__(num_warmup_steps, num_delay_steps, *args, **kwargs)

    def value_lambda(self, current_step: int) -> float:
        """Lambda function for linearly decreasing values with linear warmup.

        :param current_step: Current step
        :type current_step: int
        :return: Value at the current step
        :rtype: float
        """
        if self._check_delay(current_step):
            return 0.0
        if self._check_warmup(current_step):
            return (current_step - self.num_delay_steps) / self.num_warmup_steps
        current_step = current_step - self.num_delay_steps - self.num_warmup_steps
        remaining_steps = self.num_training_steps - self.num_delay_steps - self.num_warmup_steps
        step_size = (1 - self.final_value) / remaining_steps
        return max(self.final_value, 1 - step_size * current_step)


class ConstantSchedulerWithLinearWarmup(LambdaWarmupScheduler):
    def value_lambda(self, current_step: int) -> float:
        """Lambda function for no decay with linear warmup.

        :param current_step: Current step
        :type current_step: int
        :return: Value at the current step
        :rtype: float
        """
        if self._check_delay(current_step):
            return 0.0
        if self._check_warmup(current_step):
            return (current_step - self.num_delay_steps) / self.num_warmup_steps
        return 1.0


class ConstantSchedulerWithQuadraticWarmup(LambdaWarmupScheduler):
    def value_lambda(self, current_step: int) -> float:
        """Lambda function for no decay with quadratic warmup.

        :param current_step: Current step
        :type current_step: int
        :return: Value at the current step
        :rtype: float
        """
        if self._check_delay(current_step):
            return 0.0
        if self._check_warmup(current_step):
            return ((current_step - self.num_delay_steps) / self.num_warmup_steps) ** 2
        return 1.0


class GenericScheduler(Callback, ABC):

    def __init__(self, *args, keys: Sequence[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if keys is None:
            raise ValueError("keys must be provided")
        self.keys = keys
        self.values: Dict[str, float] = {}

    def step(self, key: str, current_step: int) -> float:
        value = self.values[key]
        return value * self.value_lambda(current_step)

    @abstractmethod
    def value_lambda(self, current_step: int) -> float: ...

    def get_value(self, sub_keys: Sequence[str], obj: object) -> object:
        for sub_key in sub_keys:
            try:
                obj = obj[int(sub_key)]
            except ValueError:
                obj = getattr(obj, sub_key)
        return obj

    def set_value(self, sub_keys: Sequence[str], obj: object, value: float) -> None:
        obj = self.get_value(sub_keys[:-1], obj)
        setattr(obj, sub_keys[-1], value)

    def on_train_start(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        for key in self.keys:
            sub_keys = key.split(".")
            self.values[key] = float(self.get_value(sub_keys, pl_module))

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        step = trainer.global_step + 1
        for key in self.keys:
            value = self.step(key, step)
            sub_keys = key.split(".")
            self.set_value(sub_keys, pl_module, value)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for key in self.keys:
            value = self.values[key]
            sub_keys = key.split(".")
            self.set_value(sub_keys, pl_module, value)


class GenericLinearSchedulerWithLinearWarmup(LinearSchedulerWithLinearWarmup, GenericScheduler):
    pass


class GenericConstantSchedulerWithLinearWarmup(ConstantSchedulerWithLinearWarmup, GenericScheduler):
    pass


class GenericConstantSchedulerWithQuadraticWarmup(ConstantSchedulerWithQuadraticWarmup, GenericScheduler):
    pass
