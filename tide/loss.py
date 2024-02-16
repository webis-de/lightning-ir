from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch


PAD_VALUE = -10000
EPS = 1e-6


def custom_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_x, _ = x.max(dim=dim, keepdim=True)
    x = x - max_x
    exp_x = x.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class LossFunction(ABC):
    def __init__(
        self,
        reduction: Optional[Literal["mean", "sum"]] = "mean",
        in_batch_loss: Literal["ce", "hinge"] | None = None,
        # TODO add multiple in batch losses (hinge)
    ):
        self.reduction = reduction
        self.in_batch_loss = in_batch_loss

    @abstractmethod
    def compute_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def aggregate(
        self,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.reduction is None:
            return loss
        if mask is not None:
            loss = loss[~mask]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unknown reduction {self.reduction}")

    def in_batch_cross_entropy(self, scores: torch.Tensor) -> torch.Tensor:
        labels = torch.arange(scores.shape[0], device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, labels, reduction="none")
        return self.aggregate(loss)

    def in_batch_hinge(self, scores: torch.Tensor) -> torch.Tensor:
        labels = torch.eye(scores.shape[0], device=scores.device) * 2 - 1
        scores = 1 - scores
        loss = torch.nn.functional.hinge_embedding_loss(
            scores, labels, reduction="none"
        )
        return self.aggregate(loss)

    def compute_in_batch_loss(self, scores: torch.Tensor) -> torch.Tensor:
        if self.in_batch_loss is None:
            return torch.tensor(0.0, requires_grad=True, device=scores.device)
        if self.in_batch_loss == "ce":
            return self.in_batch_cross_entropy(scores)
        if self.in_batch_loss == "hinge":
            return self.in_batch_hinge(scores)
        raise ValueError(f"Unknown in batch loss {self.in_batch_loss}")


class MarginMSE(LossFunction):
    def compute_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = scores.eq(PAD_VALUE) | labels.eq(PAD_VALUE)
        logit_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
        label_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
        loss = torch.nn.functional.mse_loss(logit_diff, label_diff, reduction="none")
        mask = ~torch.triu(~mask[..., None].expand_as(loss), diagonal=1)
        return self.aggregate(loss, mask)


class RankNet(LossFunction):
    def __init__(
        self,
        reduction: Literal["mean", "sum"] | None = "mean",
        in_batch_loss: Literal["ce", "hinge"] | None = None,
        discounted: bool = False,
    ):
        super().__init__(reduction, in_batch_loss)
        self.discounted = discounted

    def compute_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        greater = labels[..., None] > labels[:, None]
        scores_mask = scores.eq(PAD_VALUE)
        label_mask = labels.eq(PAD_VALUE)
        mask = scores_mask[..., None] | label_mask[..., None] | ~greater
        diff = scores[..., None] - scores[:, None]
        weight = None
        if self.discounted:
            ranks = torch.argsort(labels, descending=True) + 1
            discounts = 1 / torch.log2(ranks + 1)
            weight = torch.max(discounts[..., None], discounts[:, None])
            weight = weight.masked_fill(mask, 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff, greater.to(diff), reduction="none", weight=weight
        )
        loss = loss.masked_fill(mask, 0)
        return self.aggregate(loss, mask)


class LocalizedContrastive(LossFunction):
    def compute_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (scores == PAD_VALUE)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        labels = labels.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(scores, labels, reduction="none")
        loss = loss[:, None]
        return self.aggregate(loss)


class KLDivergence(LossFunction):
    def compute_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (scores == PAD_VALUE)
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        labels = torch.nn.functional.log_softmax(labels.to(scores), dim=-1)
        loss = torch.nn.functional.kl_div(
            scores, labels.to(scores), reduction="none", log_target=True
        )
        return self.aggregate(loss, mask)


class RankHinge(LossFunction):
    def compute_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = 1 - scores
        greater = labels[..., None] > labels[:, None]
        scores_mask = scores.eq(PAD_VALUE)
        label_mask = labels.eq(PAD_VALUE)
        mask = scores_mask[..., None] | label_mask[..., None] | ~greater
        diff = scores[..., None] - scores[:, None]
        loss = diff.masked_fill(mask, 0).clamp(min=0)
        return self.aggregate(loss, mask)
