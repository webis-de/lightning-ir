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
        in_batch_negatives: bool = False,
    ):
        self.reduction = reduction
        self.in_batch_negatives = in_batch_negatives

    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor: ...

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

    def compute_in_batch_negative_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # softmax cross entropy over in-batch negatives
        batch_size, num_docs = logits.shape
        num_ib_docs = num_docs // batch_size
        num_ob_docs = num_docs - num_ib_docs
        mask = logits.eq(PAD_VALUE)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        logits = logits - logits.max(dim=1, keepdim=True).values
        exp_logits = logits.exp().view(-1)
        ib_idcs = (
            (torch.arange(batch_size).repeat_interleave(num_ib_docs) * num_docs)
            + torch.arange(num_ib_docs).repeat(batch_size)
            + torch.arange(batch_size)
            .multiply(num_ib_docs)
            .repeat_interleave(num_ib_docs)
        )
        all_idcs = torch.arange(logits.numel())
        ob_idcs = all_idcs[all_idcs.not_equal(ib_idcs[:, None]).all(0)]
        ib_exp_logits = exp_logits[ib_idcs].view(batch_size, num_ib_docs)
        ob_exp_logits = exp_logits[ob_idcs].view(batch_size, num_ob_docs)
        denominator = ib_exp_logits + ob_exp_logits.sum(dim=1, keepdim=True)
        loss = -torch.log(ib_exp_logits / denominator)
        mask = mask.view(-1)[ib_idcs].view(batch_size, num_ib_docs)
        return self.aggregate(loss, mask)


class MarginMSE(LossFunction):
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = logits.eq(PAD_VALUE) | labels.eq(PAD_VALUE)
        logit_diff = logits.unsqueeze(-1) - logits.unsqueeze(-2)
        label_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
        loss = torch.nn.functional.mse_loss(logit_diff, label_diff, reduction="none")
        mask = ~torch.triu(~mask[..., None].expand_as(loss), diagonal=1)
        return self.aggregate(loss, mask)


class RankNet(LossFunction):
    def __init__(
        self,
        reduction: Literal["mean", "sum"] | None = "mean",
        in_batch_negatives: bool = False,
        discounted: bool = False,
    ):
        super().__init__(reduction, in_batch_negatives)
        self.discounted = discounted

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        greater = labels[..., None] > labels[:, None]
        logits_mask = logits.eq(PAD_VALUE)
        label_mask = labels.eq(PAD_VALUE)
        mask = logits_mask[..., None] | label_mask[..., None] | ~greater
        diff = logits[..., None] - logits[:, None]
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
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        labels = labels.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        loss = loss[:, None]
        return self.aggregate(loss)


class KLDivergence(LossFunction):
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        labels = torch.nn.functional.log_softmax(labels.to(logits), dim=-1)
        loss = torch.nn.functional.kl_div(
            logits, labels.to(logits), reduction="none", log_target=True
        )
        return self.aggregate(loss, mask)
