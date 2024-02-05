from typing import Literal, Optional

import torch


PAD_VALUE = -10000
EPS = 1e-6


class LossFunction:
    def __init__(self, reduction: Optional[Literal["mean", "sum"]] = "mean"):
        self.reduction = reduction

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

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


class MarginMSE(LossFunction):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.shape[-1] == 2
        mask = ((logits == PAD_VALUE) | (labels == PAD_VALUE)).any(-1)
        logit_diff = logits.unsqueeze(-1) - logits.unsqueeze(-2)
        label_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
        loss = torch.nn.functional.mse_loss(logit_diff, label_diff, reduction="none")
        mask = ~torch.triu(~mask[:, None, None].expand_as(loss), diagonal=1)
        return self.aggregate(loss, mask)


class RankNet(LossFunction):
    def __init__(
        self,
        reduction: Literal["mean", "sum"] | None = "mean",
        discounted: bool = False,
    ):
        super().__init__(reduction)
        self.discounted = discounted

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        greater = labels[..., None] > labels[:, None]
        logits_mask = logits == PAD_VALUE
        label_mask = labels == PAD_VALUE
        mask = (
            logits_mask[..., None]
            | logits_mask[:, None]
            | label_mask[..., None]
            | label_mask[:, None]
            | ~greater
        )
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
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        labels = labels.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        return self.aggregate(loss)
