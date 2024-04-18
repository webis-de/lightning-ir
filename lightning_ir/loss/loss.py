from abc import ABC, abstractmethod
from typing import Literal, Tuple

import torch


class LossFunction(ABC):
    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "LossFunction.compute_loss must be implemented by subclasses"
        )

    def process_targets(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if targets.ndim > logits.ndim:
            return targets.max(-1).values
        return targets


class IALossFunction(LossFunction):
    def process_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets


class PairwiseLossFunction(LossFunction):
    def get_pairwise_idcs(self, targets: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # pos items are items where label is greater than other label in sample
        return torch.nonzero(targets[..., None] > targets[:, None], as_tuple=True)


class ListwiseLossFunction(LossFunction):
    pass


class MarginMSE(PairwiseLossFunction):
    def __init__(self, margin: float | Literal["scores"] = 1.0):
        self.margin = margin

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.process_targets(logits, targets)
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(targets)
        pos = logits[query_idcs, pos_idcs]
        neg = logits[query_idcs, neg_idcs]
        margin = pos - neg
        if isinstance(self.margin, float):
            target_margin = torch.tensor(self.margin, device=logits.device)
        elif self.margin == "scores":
            target_margin = (
                targets[query_idcs, pos_idcs] - targets[query_idcs, neg_idcs]
            )
        else:
            raise ValueError("invalid margin type")
        loss = torch.nn.functional.mse_loss(margin, target_margin.clamp(min=0))
        return loss


class ConstantMarginMSE(MarginMSE):
    def __init__(self, margin: float = 1.0):
        super().__init__(margin)


class SupervisedMarginMSE(MarginMSE):
    def __init__(self):
        super().__init__("scores")


class RankNet(PairwiseLossFunction):
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.process_targets(logits, targets)
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(targets)
        pos = logits[query_idcs, pos_idcs]
        neg = logits[query_idcs, neg_idcs]
        margin = pos - neg
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            margin, torch.ones_like(margin)
        )
        return loss


class KLDivergence(ListwiseLossFunction):
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.process_targets(logits, targets)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        targets = torch.nn.functional.log_softmax(targets.to(logits), dim=-1)
        loss = torch.nn.functional.kl_div(logits, targets, log_target=True)
        return loss


class InBatchLossFunction(LossFunction):
    def __init__(
        self,
        pos_sampling_technique: Literal["all", "first"] = "all",
        neg_sampling_technique: Literal["all", "first"] = "all",
    ):
        super().__init__()
        self.pos_sampling_technique = pos_sampling_technique
        self.neg_sampling_technique = neg_sampling_technique

    def get_ib_masks(
        self, batch_size: int, depth: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        min_idx = torch.arange(batch_size)[:, None] * depth
        max_idx = min_idx + depth
        if self.pos_sampling_technique == "all":
            pos_mask = torch.arange(batch_size * depth)[None].greater_equal(
                min_idx
            ) & torch.arange(batch_size * depth)[None].less(max_idx)
        elif self.pos_sampling_technique == "first":
            pos_mask = torch.arange(batch_size * depth)[None].eq(min_idx)
        else:
            raise ValueError("invalid pos sampling technique")
        if self.neg_sampling_technique == "all":
            neg_mask = torch.arange(batch_size * depth)[None].less(
                min_idx
            ) | torch.arange(batch_size * depth)[None].greater_equal(max_idx)
        elif self.neg_sampling_technique == "first":
            neg_mask = torch.arange(batch_size * depth)[None, None].eq(min_idx).any(
                1
            ) & torch.arange(batch_size * depth)[None].ne(min_idx)
        else:
            raise ValueError("invalid neg sampling technique")
        return pos_mask, neg_mask

    def compute_loss(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "InBatchLossFunction.compute_loss must be implemented by subclasses"
        )


class InBatchCrossEntropy(InBatchLossFunction):
    def compute_loss(self, logits: torch.Tensor) -> torch.Tensor:
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss
