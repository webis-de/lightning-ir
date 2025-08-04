"""
Loss functions for the Lightning IR framework.

This module defines various loss functions used in the Lightning IR framework for training and evaluating models.
It includes loss functions for scoring, embedding, pairwise, listwise, and in-batch losses, as well as regularization
losses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Tuple

import torch

if TYPE_CHECKING:
    from ..base import LightningIROutput
    from ..bi_encoder import BiEncoderOutput
    from ..data import TrainBatch


class LossFunction(ABC):
    """Base class for loss functions in the Lightning IR framework."""

    @abstractmethod
    def compute_loss(self, output: LightningIROutput, *args, **kwargs) -> torch.Tensor:
        """Compute the loss for the given output.

        Args:
            output (LightningIROutput): The output from the model.
        Returns:
            torch.Tensor: The computed loss.
        """
        ...

    def process_scores(self, output: LightningIROutput) -> torch.Tensor:
        """Process the scores from the output.

        Args:
            output (LightningIROutput): The output from the model.
        Returns:
            torch.Tensor: The scores tensor.
        """
        if output.scores is None:
            raise ValueError("Expected scores in LightningIROutput")
        return output.scores

    def process_targets(self, scores: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        """Process the targets from the batch.

        Args:
            scores (torch.Tensor): The scores tensor.
            batch (TrainBatch): The training batch.
        Returns:
            torch.Tensor: The processed targets tensor.
        """
        targets = batch.targets
        if targets is None:
            raise ValueError("Expected targets in TrainBatch")
        if targets.ndim > scores.ndim:
            return targets.amax(-1)
        return targets


class ScoringLossFunction(LossFunction):
    """Base class for loss functions that operate on scores."""

    @abstractmethod
    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the loss based on the scores and targets in the output and batch.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        ...


class EmbeddingLossFunction(LossFunction):
    """Base class for loss functions that operate on embeddings."""

    @abstractmethod
    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the loss based on the embeddings in the output.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        ...


class PairwiseLossFunction(ScoringLossFunction):
    """Base class for pairwise loss functions."""

    def get_pairwise_idcs(self, targets: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Get pairwise indices for positive and negative samples based on targets.

        Args:
            targets (torch.Tensor): The targets tensor containing relevance labels.
        Returns:
            Tuple[torch.Tensor, ...]: Indices of positive and negative samples.
        """
        # positive items are items where label is greater than other label in sample
        return torch.nonzero(targets[..., None] > targets[:, None], as_tuple=True)


class ListwiseLossFunction(ScoringLossFunction):
    """Base class for listwise loss functions."""

    pass


class MarginMSE(PairwiseLossFunction):
    """Mean Squared Error loss with a margin for pairwise ranking tasks.
    Originally proposed in: `Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation \
    <https://arxiv.org/abs/2010.02666>`_
    """

    def __init__(self, margin: float | Literal["scores"] = 1.0):
        """Initialize the MarginMSE loss function.

        Args:
            margin (float | Literal["scores"]): The margin value for the loss.
        """
        self.margin = margin

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the MarginMSE loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        Raises:
            ValueError: If the margin type is invalid.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(targets)
        pos = scores[query_idcs, pos_idcs]
        neg = scores[query_idcs, neg_idcs]
        margin = pos - neg
        if isinstance(self.margin, float):
            target_margin = torch.tensor(self.margin, device=scores.device)
        elif self.margin == "scores":
            target_margin = targets[query_idcs, pos_idcs] - targets[query_idcs, neg_idcs]
        else:
            raise ValueError("invalid margin type")
        loss = torch.nn.functional.mse_loss(margin, target_margin)
        return loss


class ConstantMarginMSE(MarginMSE):
    """Constant Margin MSE loss for pairwise ranking tasks with a fixed margin."""

    def __init__(self, margin: float = 1.0):
        """Initialize the ConstantMarginMSE loss function.

        Args:
            margin (float): The fixed margin value for the loss.
        """
        super().__init__(margin)


class SupervisedMarginMSE(MarginMSE):
    """Supervised Margin MSE loss for pairwise ranking tasks with a dynamic margin."""

    def __init__(self):
        """Initialize the SupervisedMarginMSE loss function."""
        super().__init__("scores")


class RankNet(PairwiseLossFunction):
    """RankNet loss function for pairwise ranking tasks.
    Originally proposed in: `Learning to Rank using Gradient Descent \
    <https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_
    """

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the RankNet loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(targets)
        pos = scores[query_idcs, pos_idcs]
        neg = scores[query_idcs, neg_idcs]
        margin = pos - neg
        loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        return loss


class KLDivergence(ListwiseLossFunction):
    """Kullback-Leibler Divergence loss for listwise ranking tasks.
    Originally proposed in: `On Information and Sufficiency \
    <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full>`_
    """

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the Kullback-Leibler Divergence loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        targets = torch.nn.functional.log_softmax(targets.to(scores), dim=-1)
        loss = torch.nn.functional.kl_div(scores, targets, log_target=True, reduction="batchmean")
        return loss


class PearsonCorrelation(ListwiseLossFunction):
    """Pearson Correlation loss for listwise ranking tasks.
    Originally proposed in: `Note on regression and inheritance in the case of two parents \
    <https://royalsocietypublishing.org/doi/10.1098/rspl.1895.0041>`_"""

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the Pearson Correlation loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch).to(scores)
        centered_scores = scores - scores.mean(dim=-1, keepdim=True)
        centered_targets = targets - targets.mean(dim=-1, keepdim=True)
        pearson = torch.nn.functional.cosine_similarity(centered_scores, centered_targets, dim=-1)
        loss = (1 - pearson).mean()
        return loss


class InfoNCE(ListwiseLossFunction):
    """InfoNCE loss for listwise ranking tasks.
    Originally proposed in: `Representation Learning with Contrastive Predictive Coding \
    <https://arxiv.org/abs/1807.03748>`_
    """

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the InfoNCE loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        targets = targets.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss


class ApproxLossFunction(ListwiseLossFunction):
    """Base class for approximate loss functions that compute ranks from scores."""

    def __init__(self, temperature: float = 1) -> None:
        """Initialize the ApproxLossFunction.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
        """
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def get_approx_ranks(scores: torch.Tensor, temperature: float) -> torch.Tensor:
        """Compute approximate ranks from scores.

        Args:
            scores (torch.Tensor): The input scores.
            temperature (float): Temperature parameter for scaling the scores.
        Returns:
            torch.Tensor: The computed approximate ranks.
        """
        score_diff = scores[:, None] - scores[..., None]
        normalized_score_diff = torch.sigmoid(score_diff / temperature)
        # set diagonal to 0
        normalized_score_diff = normalized_score_diff * (1 - torch.eye(scores.shape[1], device=scores.device))
        approx_ranks = normalized_score_diff.sum(-1) + 1
        return approx_ranks


class ApproxNDCG(ApproxLossFunction):
    """Approximate NDCG loss function for ranking tasks."""

    def __init__(self, temperature: float = 1, scale_gains: bool = True):
        """Initialize the ApproxNDCG loss function.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
            scale_gains (bool): Whether to scale the gains. Defaults to True.
        """
        super().__init__(temperature)
        self.scale_gains = scale_gains

    @staticmethod
    def get_dcg(
        ranks: torch.Tensor,
        targets: torch.Tensor,
        k: int | None = None,
        scale_gains: bool = True,
    ) -> torch.Tensor:
        """Compute the Discounted Cumulative Gain (DCG) for the given ranks and targets.

        Args:
            ranks (torch.Tensor): The ranks of the items.
            targets (torch.Tensor): The relevance scores of the items.
            k (int | None): Optional cutoff for the ranks. If provided, only computes DCG for the top k items.
            scale_gains (bool): Whether to scale the gains. Defaults to True.
        Returns:
            torch.Tensor: The computed DCG values.
        """
        log_ranks = torch.log2(1 + ranks)
        discounts = 1 / log_ranks
        if scale_gains:
            gains = 2**targets - 1
        else:
            gains = targets
        dcgs = gains * discounts
        if k is not None:
            dcgs = dcgs.masked_fill(ranks > k, 0)
        return dcgs.sum(dim=-1)

    @staticmethod
    def get_ndcg(
        ranks: torch.Tensor,
        targets: torch.Tensor,
        k: int | None = None,
        scale_gains: bool = True,
        optimal_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Normalized Discounted Cumulative Gain (NDCG) for the given ranks and targets.

        Args:
            ranks (torch.Tensor): The ranks of the items.
            targets (torch.Tensor): The relevance scores of the items.
            k (int | None): Cutoff for the ranks. If provided, only computes NDCG for the top k items. Defaults to None.
            scale_gains (bool): Whether to scale the gains. Defaults to True.
            optimal_targets (torch.Tensor | None): Optional tensor of optimal targets for normalization. If None, uses
                the targets. Defaults to None.
        Returns:
            torch.Tensor: The computed NDCG values.
        """
        targets = targets.clamp(min=0)
        if optimal_targets is None:
            optimal_targets = targets
        optimal_ranks = torch.argsort(torch.argsort(optimal_targets, descending=True))
        optimal_ranks = optimal_ranks + 1
        dcg = ApproxNDCG.get_dcg(ranks, targets, k, scale_gains)
        idcg = ApproxNDCG.get_dcg(optimal_ranks, optimal_targets, k, scale_gains)
        ndcg = dcg / (idcg.clamp(min=1e-12))
        return ndcg

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the ApproxNDCG loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        approx_ranks = self.get_approx_ranks(scores, self.temperature)
        ndcg = self.get_ndcg(approx_ranks, targets, k=None, scale_gains=self.scale_gains)
        loss = 1 - ndcg
        return loss.mean()


class ApproxMRR(ApproxLossFunction):
    """Approximate Mean Reciprocal Rank (MRR) loss function for ranking tasks."""

    def __init__(self, temperature: float = 1):
        """Initialize the ApproxMRR loss function.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
        """
        super().__init__(temperature)

    @staticmethod
    def get_mrr(ranks: torch.Tensor, targets: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Compute the Mean Reciprocal Rank (MRR) for the given ranks and targets.

        Args:
            ranks (torch.Tensor): The ranks of the items.
            targets (torch.Tensor): The relevance scores of the items.
            k (int | None): Optional cutoff for the ranks. If provided, only computes MRR for the top k items.
        Returns:
            torch.Tensor: The computed MRR values.
        """
        targets = targets.clamp(None, 1)
        reciprocal_ranks = 1 / ranks
        mrr = reciprocal_ranks * targets
        if k is not None:
            mrr = mrr.masked_fill(ranks > k, 0)
        mrr = mrr.max(dim=-1)[0]
        return mrr

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the ApproxMRR loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        approx_ranks = self.get_approx_ranks(scores, self.temperature)
        mrr = self.get_mrr(approx_ranks, targets, k=None)
        loss = 1 - mrr
        return loss.mean()


class ApproxRankMSE(ApproxLossFunction):
    """Approximate Rank Mean Squared Error (RankMSE) loss function for ranking tasks."""

    def __init__(
        self,
        temperature: float = 1,
        discount: Literal["log2", "reciprocal"] | None = None,
    ):
        """Initialize the ApproxRankMSE loss function.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
            discount (Literal["log2", "reciprocal"] | None): Discounting strategy for the loss. If None, no discounting
                is applied. Defaults to None.
        """
        super().__init__(temperature)
        self.discount = discount

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        """Compute the ApproxRankMSE loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        approx_ranks = self.get_approx_ranks(scores, self.temperature)
        ranks = torch.argsort(torch.argsort(targets, descending=True)) + 1
        loss = torch.nn.functional.mse_loss(approx_ranks, ranks.to(approx_ranks), reduction="none")
        if self.discount == "log2":
            weight = 1 / torch.log2(ranks + 1)
        elif self.discount == "reciprocal":
            weight = 1 / ranks
        else:
            weight = 1
        loss = loss * weight
        loss = loss.mean()
        return loss


class NeuralLossFunction(ListwiseLossFunction):
    """Base class for neural loss functions that compute ranks from scores using neural sorting."""

    # TODO add neural loss functions

    def __init__(self, temperature: float = 1, tol: float = 1e-5, max_iter: int = 50) -> None:
        """Initialize the NeuralLossFunction.

        Args:
            temperature (float): Temperature parameter for scaling the scores. Defaults to 1.
            tol (float): Tolerance for convergence. Defaults to 1e-5.
            max_iter (int): Maximum number of iterations for convergence. Defaults to 50.
        """
        super().__init__()
        self.temperature = temperature
        self.tol = tol
        self.max_iter = max_iter

    def neural_sort(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute the neural sort permutation matrix from scores.

        Args:
            scores (torch.Tensor): The input scores tensor.
        Returns:
            torch.Tensor: The computed permutation matrix.
        """
        # https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py
        scores = scores.unsqueeze(-1)
        dim = scores.shape[1]
        one = torch.ones((dim, 1), device=scores.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = dim + 1 - 2 * (torch.arange(dim, device=scores.device) + 1)
        C = torch.matmul(scores, scaling.to(scores).unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        P_hat = torch.nn.functional.softmax(P_max / self.temperature, dim=-1)

        P_hat = self.sinkhorn_scaling(P_hat)

        return P_hat

    def sinkhorn_scaling(self, mat: torch.Tensor) -> torch.Tensor:
        """Apply Sinkhorn scaling to the permutation matrix.

        Args:
            mat (torch.Tensor): The input permutation matrix.
        Returns:
            torch.Tensor: The scaled permutation matrix.
        """
        # https://github.com/allegro/allRank/blob/master/allrank/models/losses/loss_utils.py#L8
        idx = 0
        while True:
            if (
                torch.max(torch.abs(mat.sum(dim=2) - 1.0)) < self.tol
                and torch.max(torch.abs(mat.sum(dim=1) - 1.0)) < self.tol
            ) or idx > self.max_iter:
                break
            mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1e-12)
            mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=1e-12)
            idx += 1

        return mat

    def get_sorted_targets(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get the sorted targets based on the neural sort permutation matrix.

        Args:
            scores (torch.Tensor): The input scores tensor.
            targets (torch.Tensor): The targets tensor.
        Returns:
            torch.Tensor: The sorted targets tensor.
        """
        permutation_matrix = self.neural_sort(scores)
        pred_sorted_targets = torch.matmul(permutation_matrix, targets[..., None].to(permutation_matrix)).squeeze(-1)
        return pred_sorted_targets


class InBatchLossFunction(LossFunction):
    """Base class for in-batch loss functions that compute in-batch indices for positive and negative samples."""

    def __init__(
        self,
        pos_sampling_technique: Literal["all", "first"] = "all",
        neg_sampling_technique: Literal["all", "first", "all_and_non_first"] = "all",
        max_num_neg_samples: int | None = None,
    ):
        """Initialize the InBatchLossFunction.

        Args:
            pos_sampling_technique (Literal["all", "first"]): Technique for positive sample sampling.
            neg_sampling_technique (Literal["all", "first", "all_and_non_first"]): Technique for negative sample
                sampling.
            max_num_neg_samples (int | None): Maximum number of negative samples to consider. If None, all negative
                samples are considered.
        Raises:
            ValueError: If the negative sampling technique is invalid for the given positive sampling technique.
        """
        super().__init__()
        self.pos_sampling_technique = pos_sampling_technique
        self.neg_sampling_technique = neg_sampling_technique
        self.max_num_neg_samples = max_num_neg_samples
        if self.neg_sampling_technique == "all_and_non_first" and self.pos_sampling_technique != "first":
            raise ValueError("all_and_non_first is only valid with pos_sampling_technique first")

    def _get_pos_mask(
        self,
        num_queries: int,
        num_docs: int,
        max_idx: torch.Tensor,
        min_idx: torch.Tensor,
        output: LightningIROutput,
        batch: TrainBatch,
    ) -> torch.Tensor:
        """Get the mask for positive samples based on the sampling technique.

        Args:
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            max_idx (torch.Tensor): Maximum index for each query.
            min_idx (torch.Tensor): Minimum index for each query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: A mask tensor indicating the positions of positive samples.
        Raises:
            ValueError: If the positive sampling technique is invalid.
        """
        if self.pos_sampling_technique == "all":
            pos_mask = torch.arange(num_queries * num_docs)[None].greater_equal(min_idx) & torch.arange(
                num_queries * num_docs
            )[None].less(max_idx)
        elif self.pos_sampling_technique == "first":
            pos_mask = torch.arange(num_queries * num_docs)[None].eq(min_idx)
        else:
            raise ValueError("invalid pos sampling technique")
        return pos_mask

    def _get_neg_mask(
        self,
        num_queries: int,
        num_docs: int,
        max_idx: torch.Tensor,
        min_idx: torch.Tensor,
        output: LightningIROutput,
        batch: TrainBatch,
    ) -> torch.Tensor:
        """Get the mask for negative samples based on the sampling technique.

        Args:
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            max_idx (torch.Tensor): Maximum index for each query.
            min_idx (torch.Tensor): Minimum index for each query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: A mask tensor indicating the positions of negative samples.
        Raises:
            ValueError: If the negative sampling technique is invalid.
        """
        if self.neg_sampling_technique == "all_and_non_first":
            neg_mask = torch.arange(num_queries * num_docs)[None].not_equal(min_idx)
        elif self.neg_sampling_technique == "all":
            neg_mask = torch.arange(num_queries * num_docs)[None].less(min_idx) | torch.arange(num_queries * num_docs)[
                None
            ].greater_equal(max_idx)
        elif self.neg_sampling_technique == "first":
            neg_mask = torch.arange(num_queries * num_docs)[None, None].eq(min_idx).any(1) & torch.arange(
                num_queries * num_docs
            )[None].ne(min_idx)
        else:
            raise ValueError("invalid neg sampling technique")
        return neg_mask

    def get_ib_idcs(self, output: LightningIROutput, batch: TrainBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get in-batch indices for positive and negative samples.

        Args:
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Indices of positive and negative samples.
        Raises:
            ValueError: If scores are not present in the output.
        """
        if output.scores is None:
            raise ValueError("Expected scores in LightningIROutput")
        num_queries, num_docs = output.scores.shape
        min_idx = torch.arange(num_queries)[:, None] * num_docs
        max_idx = min_idx + num_docs
        pos_mask = self._get_pos_mask(num_queries, num_docs, max_idx, min_idx, output, batch)
        neg_mask = self._get_neg_mask(num_queries, num_docs, max_idx, min_idx, output, batch)
        pos_idcs = pos_mask.nonzero(as_tuple=True)[1]
        neg_idcs = neg_mask.nonzero(as_tuple=True)[1]
        if self.max_num_neg_samples is not None:
            neg_idcs = neg_idcs.view(num_queries, -1)
            if neg_idcs.shape[-1] > 1:
                neg_idcs = neg_idcs[:, torch.randperm(neg_idcs.shape[-1])]
            neg_idcs = neg_idcs[:, : self.max_num_neg_samples]
            neg_idcs = neg_idcs.reshape(-1)
        return pos_idcs, neg_idcs


class ScoreBasedInBatchLossFunction(InBatchLossFunction):
    """Base class for in-batch loss functions that compute in-batch indices based on scores."""

    def __init__(self, min_target_diff: float, max_num_neg_samples: int | None = None):
        """Initialize the ScoreBasedInBatchLossFunction.

        Args:
            min_target_diff (float): Minimum target difference for negative sampling.
            max_num_neg_samples (int | None): Maximum number of negative samples.
        """
        super().__init__(
            pos_sampling_technique="first",
            neg_sampling_technique="all_and_non_first",
            max_num_neg_samples=max_num_neg_samples,
        )
        self.min_target_diff = min_target_diff

    def _sort_mask(
        self, mask: torch.Tensor, num_queries: int, num_docs: int, output: LightningIROutput, batch: TrainBatch
    ) -> torch.Tensor:
        """Sort the mask based on the scores and targets.

        Args:
            mask (torch.Tensor): The initial mask tensor.
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: The sorted mask tensor.
        """
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch)
        idcs = targets.argsort(descending=True).argsort().cpu()
        idcs = idcs + torch.arange(num_queries)[:, None] * num_docs
        block_idcs = torch.arange(num_docs)[None] + torch.arange(num_queries)[:, None] * num_docs
        return mask.scatter(1, block_idcs, mask.gather(1, idcs))

    def _get_pos_mask(
        self,
        num_queries: int,
        num_docs: int,
        max_idx: torch.Tensor,
        min_idx: torch.Tensor,
        output: LightningIROutput,
        batch: TrainBatch,
    ) -> torch.Tensor:
        """Get the mask for positive samples and sort it based on scores.

        Args:
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            max_idx (torch.Tensor): Maximum index for each query.
            min_idx (torch.Tensor): Minimum index for each query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: A mask tensor indicating the positions of positive samples, sorted by scores.
        """
        pos_mask = super()._get_pos_mask(num_queries, num_docs, max_idx, min_idx, output, batch)
        pos_mask = self._sort_mask(pos_mask, num_queries, num_docs, output, batch)
        return pos_mask

    def _get_neg_mask(
        self,
        num_queries: int,
        num_docs: int,
        max_idx: torch.Tensor,
        min_idx: torch.Tensor,
        output: LightningIROutput,
        batch: TrainBatch,
    ) -> torch.Tensor:
        """Get the mask for negative samples and sort it based on scores.

        Args:
            num_queries (int): Number of queries in the batch.
            num_docs (int): Number of documents per query.
            max_idx (torch.Tensor): Maximum index for each query.
            min_idx (torch.Tensor): Minimum index for each query.
            output (LightningIROutput): The output from the model containing scores.
            batch (TrainBatch): The training batch containing targets.
        Returns:
            torch.Tensor: A mask tensor indicating the positions of negative samples, sorted by scores.
        """
        neg_mask = super()._get_neg_mask(num_queries, num_docs, max_idx, min_idx, output, batch)
        neg_mask = self._sort_mask(neg_mask, num_queries, num_docs, output, batch)
        scores = self.process_scores(output)
        targets = self.process_targets(scores, batch).cpu()
        max_score, _ = targets.max(dim=-1, keepdim=True)
        score_diff = (max_score - targets).cpu()
        score_mask = score_diff.ge(self.min_target_diff)
        block_idcs = torch.arange(num_docs)[None] + torch.arange(num_queries)[:, None] * num_docs
        neg_mask = neg_mask.scatter(1, block_idcs, score_mask)
        # num_neg_samples might be different between queries
        num_neg_samples = neg_mask.sum(dim=1)
        min_num_neg_samples = num_neg_samples.min()
        additional_neg_samples = num_neg_samples - min_num_neg_samples
        for query_idx, neg_samples in enumerate(additional_neg_samples):
            neg_idcs = neg_mask[query_idx].nonzero().squeeze(1)
            additional_neg_idcs = neg_idcs[torch.randperm(neg_idcs.shape[0])][:neg_samples]
            assert neg_mask[query_idx, additional_neg_idcs].all()
            neg_mask[query_idx, additional_neg_idcs] = False
            assert neg_mask[query_idx].sum().eq(min_num_neg_samples)
        return neg_mask


class InBatchCrossEntropy(InBatchLossFunction):
    """In-batch cross-entropy loss function for ranking tasks."""

    def compute_loss(self, output: LightningIROutput) -> torch.Tensor:
        """Compute the in-batch cross-entropy loss.

        Args:
            output (LightningIROutput): The output from the model containing scores.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss


class ScoreBasedInBatchCrossEntropy(ScoreBasedInBatchLossFunction):
    """In-batch cross-entropy loss function based on scores for ranking tasks."""

    def compute_loss(self, output: LightningIROutput) -> torch.Tensor:
        """Compute the in-batch cross-entropy loss based on scores.

        Args:
            output (LightningIROutput): The output from the model containing scores.
        Returns:
            torch.Tensor: The computed loss.
        """
        scores = self.process_scores(output)
        targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss


class RegularizationLossFunction(EmbeddingLossFunction):
    """Base class for regularization loss functions that operate on embeddings."""

    def __init__(self, query_weight: float = 1e-4, doc_weight: float = 1e-4) -> None:
        """Initialize the RegularizationLossFunction.

        Args:
            query_weight (float): Weight for the query embeddings regularization. Defaults to 1e-4.
            doc_weight (float): Weight for the document embeddings regularization. Defaults to 1e-4.
        """
        self.query_weight = query_weight
        self.doc_weight = doc_weight

    def process_embeddings(self, output: BiEncoderOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process the embeddings from the output.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The processed query and document embeddings.
        Raises:
            ValueError: If query_embeddings are not present in the output.
            ValueError: If doc_embeddings are not present in the output.
        """
        query_embeddings = output.query_embeddings
        doc_embeddings = output.doc_embeddings
        if query_embeddings is None:
            raise ValueError("Expected query_embeddings in BiEncoderOutput")
        if doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")
        return query_embeddings.embeddings, doc_embeddings.embeddings


class L2Regularization(RegularizationLossFunction):
    """L2 Regularization loss function for query and document embeddings."""

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the L2 regularization loss.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        query_embeddings, doc_embeddings = self.process_embeddings(output)
        query_loss = self.query_weight * query_embeddings.norm(dim=-1).mean()
        doc_loss = self.doc_weight * doc_embeddings.norm(dim=-1).mean()
        loss = query_loss + doc_loss
        return loss


class L1Regularization(RegularizationLossFunction):
    """L1 Regularization loss function for query and document embeddings."""

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the L1 regularization loss.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        query_embeddings, doc_embeddings = self.process_embeddings(output)
        query_loss = self.query_weight * query_embeddings.norm(p=1, dim=-1).mean()
        doc_loss = self.doc_weight * doc_embeddings.norm(p=1, dim=-1).mean()
        loss = query_loss + doc_loss
        return loss


class FLOPSRegularization(RegularizationLossFunction):
    """FLOPS Regularization loss function for query and document embeddings."""

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        """Compute the FLOPS regularization loss.

        Args:
            output (BiEncoderOutput): The output from the model containing query and document embeddings.
        Returns:
            torch.Tensor: The computed loss.
        """
        query_embeddings, doc_embeddings = self.process_embeddings(output)
        query_loss = torch.sum(torch.mean(torch.abs(query_embeddings), dim=0) ** 2)
        doc_loss = torch.sum(torch.mean(torch.abs(doc_embeddings), dim=0) ** 2)
        loss = self.query_weight * query_loss + self.doc_weight * doc_loss
        return loss
