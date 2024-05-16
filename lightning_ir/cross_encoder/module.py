from typing import Any, Dict, Sequence

import torch

from ..data.data import CrossEncoderRunBatch
from ..loss.loss import InBatchLossFunction, LossFunction
from ..module import LightningIRModule
from ..tokenizer.tokenizer import CrossEncoderTokenizer
from .model import CrossEncoderConfig, CrossEncoderModel


class CrossEncoderModule(LightningIRModule):
    config_class = CrossEncoderConfig

    def __init__(
        self,
        model: CrossEncoderModel,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__(model, loss_functions, evaluation_metrics)

    def forward(self, batch: CrossEncoderRunBatch) -> torch.Tensor:
        logits = self.model.forward(
            batch.encoding.input_ids,
            batch.encoding.get("attention_mask", None),
            batch.encoding.get("token_type_ids", None),
        )
        return logits

    def predict_step(self, batch: CrossEncoderRunBatch, *args, **kwargs) -> Any:
        if isinstance(batch, CrossEncoderRunBatch):
            return self.forward(batch)
        raise ValueError(f"Unknown batch type {type(batch)}")

    def compute_losses(
        self,
        batch: CrossEncoderRunBatch,
        loss_functions: Sequence[LossFunction] | None,
    ) -> Dict[str, torch.Tensor]:
        if loss_functions is None:
            if self.loss_functions is None:
                raise ValueError("Loss function is not set")
            loss_functions = self.loss_functions
        logits = self.forward(batch)
        logits = logits.view(len(batch.query_ids), -1)
        targets = batch.targets.view(*logits.shape, -1)
        losses = {}
        for loss_function in loss_functions:
            if isinstance(loss_function, InBatchLossFunction):
                raise NotImplementedError(
                    "InBatchLossFunction not implemented for cross-encoders"
                )
            losses[loss_function.__class__.__name__] = loss_function.compute_loss(
                logits, targets
            )
        return losses
