from typing import Dict, Sequence

import torch

from ..base.module import LightningIRModule
from ..data import RankBatch, TrainBatch
from ..loss.loss import InBatchLossFunction, LossFunction
from .config import CrossEncoderConfig
from .model import CrossEncoderModel, CrossEncoderOutput
from .tokenizer import CrossEncoderTokenizer


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
        self.tokenizer: CrossEncoderTokenizer

    def forward(self, batch: RankBatch) -> CrossEncoderOutput:
        queries = batch.queries
        docs = [d for docs in batch.docs for d in docs]
        num_docs = [len(docs) for docs in batch.docs]
        encoding = self.tokenizer.tokenize(
            queries,
            docs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            num_docs=num_docs,
        )
        output = self.model.forward(**encoding)
        return output

    def predict_step(self, batch: RankBatch, *args, **kwargs) -> CrossEncoderOutput:
        if isinstance(batch, RankBatch):
            return self.forward(batch)
        raise ValueError(f"Unknown batch type {batch.__class__}")

    def compute_losses(
        self,
        batch: TrainBatch,
        loss_functions: Sequence[LossFunction] | None,
    ) -> Dict[str, torch.Tensor]:
        if loss_functions is None:
            if self.loss_functions is None:
                raise ValueError("Loss functions are not set")
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
