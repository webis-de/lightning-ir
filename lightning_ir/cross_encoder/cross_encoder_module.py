"""
Module module for cross-encoder models.

This module defines the Lightning IR module class used to implement cross-encoder models.
"""

from typing import Any, List, Mapping, Sequence, Tuple

import torch

from ..base.module import LightningIRModule
from ..data import RankBatch, SearchBatch, TrainBatch
from ..loss.loss import LossFunction, ScoringLossFunction
from .cross_encoder_config import CrossEncoderConfig
from .cross_encoder_model import CrossEncoderModel, CrossEncoderOutput
from .cross_encoder_tokenizer import CrossEncoderTokenizer


class CrossEncoderModule(LightningIRModule):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        model: CrossEncoderModel | None = None,
        loss_functions: Sequence[LossFunction | Tuple[LossFunction, float]] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        """:class:`.LightningIRModule` for cross-encoder models. It contains a :class:`.CrossEncoderModel` and a
        :class:`.CrossEncoderTokenizer` and implements the training, validation, and testing steps for the model.

        .. _ir-measures: https://ir-measur.es/en/latest/index.html

        Args:
            model_name_or_path (str | None): Name or path of backbone model or fine-tuned Lightning IR model.
                Defaults to None.
            config (CrossEncoderConfig | None): CrossEncoderConfig to apply when loading from backbone model.
                Defaults to None.
            model (CrossEncoderModel | None): Already instantiated CrossEncoderModel. Defaults to None.
            loss_functions (Sequence[LossFunction | Tuple[LossFunction, float]] | None):
                Loss functions to apply during fine-tuning, optional loss weights can be provided per loss function.
                Defaults to None.
            evaluation_metrics (Sequence[str] | None): Metrics corresponding to ir-measures_ measure strings to apply
                during validation or testing. Defaults to None.
            model_kwargs (Mapping[str, Any] | None): Additional keyword arguments to pass to `from_pretrained` when
                loading a model. Defaults to None.
        """
        super().__init__(model_name_or_path, config, model, loss_functions, evaluation_metrics, model_kwargs)
        self.model: CrossEncoderModel
        self.config: CrossEncoderConfig
        self.tokenizer: CrossEncoderTokenizer

    def forward(self, batch: RankBatch | TrainBatch | SearchBatch) -> CrossEncoderOutput:
        """Runs a forward pass of the model on a batch of data and returns the contextualized embeddings from the
        backbone model as well as the relevance scores.

        Args:
            batch (RankBatch | TrainBatch | SearchBatch): Batch of data to run the forward pass on.
        Returns:
            CrossEncoderOutput: Output of the model.
        Raises:
            ValueError: If the batch is a SearchBatch.
        """
        if isinstance(batch, SearchBatch):
            raise ValueError("Searching is not available for cross-encoders")
        queries = batch.queries
        docs = [d for docs in batch.docs for d in docs]
        num_docs = [len(docs) for docs in batch.docs]
        encoding = self.prepare_input(queries, docs, num_docs)
        output = self.model.forward(encoding["encoding"])
        return output

    def _compute_losses(self, batch: TrainBatch, output: CrossEncoderOutput) -> List[torch.Tensor]:
        """Computes the losses for a training batch."""
        if self.loss_functions is None:
            raise ValueError("loss_functions must be set in the module")

        output.scores = output.scores.view(len(batch.query_ids), -1)
        batch.targets = batch.targets.view(*output.scores.shape, -1)

        losses = []
        for loss_function, _ in self.loss_functions:
            if not isinstance(loss_function, ScoringLossFunction):
                raise RuntimeError(f"Loss function {loss_function} is not a scoring loss function")
            losses.append(loss_function.compute_loss(output, batch))
        return losses
