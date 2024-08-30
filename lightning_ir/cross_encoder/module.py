from typing import List, Sequence, Tuple

import torch

from ..base.module import LightningIRModule
from ..data import RankBatch, SearchBatch, TrainBatch
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
        loss_functions: Sequence[LossFunction | Tuple[LossFunction, float]] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__(model_name_or_path, config, model, loss_functions, evaluation_metrics)
        self.model: CrossEncoderModel
        self.config: CrossEncoderConfig
        self.tokenizer: CrossEncoderTokenizer

    def forward(self, batch: RankBatch | TrainBatch | SearchBatch) -> CrossEncoderOutput:
        if isinstance(batch, SearchBatch):
            raise NotImplementedError("Searching is not available for cross-encoders")
        queries = batch.queries
        docs = [d for docs in batch.docs for d in docs]
        num_docs = [len(docs) for docs in batch.docs]
        encoding = self.prepare_input(queries, docs, num_docs)
        output = self.model.forward(encoding["encoding"])
        return output

    def compute_losses(self, batch: TrainBatch) -> List[torch.Tensor]:
        if self.loss_functions is None:
            raise ValueError("loss_functions must be set in the module")
        output = self.forward(batch)
        scores = output.scores
        if scores is None or batch.targets is None:
            raise ValueError("scores and targets must be set in the output and batch")

        scores = scores.view(len(batch.query_ids), -1)
        targets = batch.targets.view(*scores.shape, -1)

        losses = []
        for loss_function, _ in self.loss_functions:
            if isinstance(loss_function, InBatchLossFunction):
                raise NotImplementedError("InBatchLossFunction not implemented for cross-encoders")
            losses.append(loss_function.compute_loss(scores, targets))
        return losses
