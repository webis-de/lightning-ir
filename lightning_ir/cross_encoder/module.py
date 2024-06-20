from typing import Dict, Sequence

import torch

from ..base.module import LightningIRModule
from ..data import CrossEncoderRunBatch
from ..loss.loss import InBatchLossFunction, LossFunction
from . import CrossEncoderConfig, CrossEncoderModel, CrossEncoderOutput


class CrossEncoderModule(LightningIRModule):

    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        model: CrossEncoderModel | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__(
            model_name_or_path, config, model, loss_functions, evaluation_metrics
        )
        self.model: CrossEncoderModel
        self.config: CrossEncoderConfig

    def forward(self, batch: CrossEncoderRunBatch) -> CrossEncoderOutput:
        output = self.model.forward(batch.encoding)
        return output

    def predict_step(
        self, batch: CrossEncoderRunBatch, *args, **kwargs
    ) -> CrossEncoderOutput:
        if isinstance(batch, CrossEncoderRunBatch):
            return self.forward(batch)
        raise ValueError(f"Unknown batch type {batch.__class__}")

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
