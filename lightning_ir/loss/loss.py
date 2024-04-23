from abc import ABC, abstractmethod
from typing import Literal, Tuple

import torch


def get_dcg(
    ranks: torch.Tensor,
    labels: torch.Tensor,
    k: int | None = None,
    scale_gains: bool = True,
) -> torch.Tensor:
    log_ranks = torch.log2(1 + ranks)
    discounts = 1 / log_ranks
    if scale_gains:
        gains = 2**labels - 1
    else:
        gains = labels
    dcgs = gains * discounts
    if k is not None:
        dcgs = dcgs.masked_fill(ranks > k, 0)
    return dcgs.sum(dim=-1)


def get_ndcg(
    ranks: torch.Tensor,
    labels: torch.Tensor,
    k: int | None = None,
    scale_gains: bool = True,
    optimal_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    if optimal_labels is None:
        optimal_labels = labels
    optimal_ranks = torch.argsort(torch.argsort(optimal_labels, descending=True))
    optimal_ranks = optimal_ranks + 1
    dcg = get_dcg(ranks, labels, k, scale_gains)
    idcg = get_dcg(optimal_ranks, optimal_labels, k, scale_gains)
    ndcg = dcg / (idcg.clamp(min=1e-12))
    return ndcg


def get_mrr(
    ranks: torch.Tensor, labels: torch.Tensor, k: int | None = None
) -> torch.Tensor:
    labels = labels.clamp(None, 1)
    reciprocal_ranks = 1 / ranks
    mrr = reciprocal_ranks * labels
    if k is not None:
        mrr = mrr.masked_fill(ranks > k, 0)
    mrr = mrr.max(dim=-1)[0]
    return mrr


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


class LocalizedContrastiveEstimation(ListwiseLossFunction):
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.process_targets(logits, targets)
        targets = targets.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss


class ApproxLossFunction(ListwiseLossFunction):
    def __init__(self, temperature: float = 1) -> None:
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def get_approx_ranks(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        score_diff = logits[:, None] - logits[..., None]
        normalized_score_diff = torch.sigmoid(score_diff / temperature)
        # set diagonal to 0
        normalized_score_diff = normalized_score_diff * (
            1 - torch.eye(logits.shape[1], device=logits.device)
        )
        approx_ranks = normalized_score_diff.sum(-1) + 1
        return approx_ranks


class ApproxNDCG(ApproxLossFunction):
    def __init__(self, temperature: float = 1, scale_gains: bool = True):
        super().__init__(temperature)
        self.scale_gains = scale_gains

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        approx_ranks = self.get_approx_ranks(logits, self.temperature)
        ndcg = get_ndcg(approx_ranks, labels, k=None, scale_gains=self.scale_gains)
        loss = 1 - ndcg
        return loss.mean()


class ApproxMRR(ApproxLossFunction):
    def __init__(self, temperature: float = 1):
        super().__init__(temperature)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        approx_ranks = self.get_approx_ranks(logits, self.temperature)
        mrr = get_mrr(approx_ranks, labels, k=None)
        loss = 1 - mrr
        return loss.mean()


class ApproxRankMSE(ApproxLossFunction):
    def __init__(self, temperature: float = 1, discounted: bool = False):
        super().__init__(temperature)
        self.discounted = discounted

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        approx_ranks = self.get_approx_ranks(logits, self.temperature)
        ranks = torch.argsort(labels, descending=True) + 1
        loss = torch.nn.functional.mse_loss(
            approx_ranks, ranks.to(approx_ranks), reduction="none"
        )
        if self.discounted:
            weight = 1 / torch.log2(ranks + 1)
            loss = loss * weight
        loss = loss.mean()
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
