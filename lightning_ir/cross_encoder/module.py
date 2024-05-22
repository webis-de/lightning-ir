from typing import Any, Dict, Sequence

import torch

from ..data.data import CrossEncoderRunBatch
from ..loss.loss import InBatchLossFunction, LossFunction
from ..module import LightningIRModule
from .model import CrossEncoderConfig, CrossEncoderModel, CrossEncoderOuput


class CrossEncoderModule(LightningIRModule):
    config_class = CrossEncoderConfig

    def __init__(
        self,
        model: CrossEncoderModel,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__(model, loss_functions, evaluation_metrics)
        self.model: CrossEncoderModel

    def forward(self, batch: CrossEncoderRunBatch) -> CrossEncoderOuput:
        output = self.model.forward(
            batch.encoding.input_ids,
            batch.encoding.get("attention_mask", None),
            batch.encoding.get("token_type_ids", None),
        )
        return output

    def predict_step(
        self, batch: CrossEncoderRunBatch, *args, **kwargs
    ) -> CrossEncoderOuput:
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
        output = self.forward(batch)
        scores = output.scores
        if scores is None or batch.targets is None:
            raise ValueError("scores and targets must be set in the output and batch")

        scores = scores.view(len(batch.query_ids), -1)
        targets = batch.targets.view(*scores.shape, -1)

        losses = {}
        for loss_function in loss_functions:
            if isinstance(loss_function, InBatchLossFunction):
                raise NotImplementedError(
                    "InBatchLossFunction not implemented for cross-encoders"
                )
            losses[loss_function.__class__.__name__] = loss_function.compute_loss(
                scores, targets
            )
        return losses
