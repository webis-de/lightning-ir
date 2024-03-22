from abc import ABC, abstractmethod
from typing import Dict, Literal

import torch

from .mvr import ScoringFunction


class LossFunction(ABC):
    def __init__(self, in_batch_loss: Literal["hinge", "cross-entropy"] | None = None):
        self.scoring_function: ScoringFunction
        self.in_batch_loss = in_batch_loss

    def set_scoring_function(self, scoring_function: ScoringFunction):
        self.scoring_function = scoring_function

    def compute_in_batch_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.in_batch_loss == "hinge":
            ib_loss = self.compute_in_batch_hinge_loss(
                query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
            )
        elif self.in_batch_loss == "cross-entropy":
            ib_loss = self.compute_in_batch_cross_entropy_loss(
                query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
            )
        else:
            raise ValueError("in_batch_loss must be set to 'hinge' or 'cross-entropy'")
        return {"ib_loss": ib_loss}

    @abstractmethod
    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def format_scores_all_in_sample_positives(
        self, num_queries: int, num_docs: int, scores: torch.Tensor
    ) -> torch.Tensor:
        scores = scores.view(num_queries, num_docs * num_queries)
        min_idx = torch.arange(num_queries)[:, None] * num_docs
        max_idx = min_idx + num_docs
        pos_mask = torch.arange(num_queries * num_docs)[None].greater_equal(
            min_idx
        ) & torch.arange(num_queries * num_docs)[None].less(max_idx)
        pos_scores = scores[pos_mask].view(-1, 1)
        neg_scores = (
            scores[~pos_mask].view(num_queries, -1).repeat_interleave(num_docs, 0)
        )
        scores = torch.cat((pos_scores, neg_scores), dim=1)
        return scores

    def format_scores_first_in_sample_positive(
        self, num_queries: int, num_docs: int, scores: torch.Tensor
    ) -> torch.Tensor:
        scores = scores.view(num_queries, num_docs * num_queries)
        idx = torch.arange(num_queries)[:, None] * num_docs
        pos_mask = torch.arange(num_queries * num_docs)[None].eq(idx)
        pos_scores = scores[pos_mask].view(num_queries, 1)
        neg_scores = scores[~pos_mask].view(num_queries, -1)
        scores = torch.cat((pos_scores, neg_scores), dim=1)
        return scores

    def format_scores(
        self, num_queries: int, num_docs: int, scores: torch.Tensor
    ) -> torch.Tensor:
        scores = scores.view(num_queries, num_docs * num_queries)
        min_idx = torch.arange(num_queries)[:, None] * num_docs
        max_idx = min_idx + num_docs
        pos_mask = torch.arange(num_queries * num_docs)[None].eq(min_idx)
        neg_mask = torch.arange(num_queries * num_docs)[None].less(
            min_idx
        ) | torch.arange(num_queries * num_docs)[None].greater_equal(max_idx)
        pos_scores = scores[pos_mask].view(num_queries, 1)
        neg_scores = scores[neg_mask].view(num_queries, -1)
        scores = torch.cat((pos_scores, neg_scores), dim=1)
        return scores

    def compute_in_batch_cross_entropy_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_queries = query_embeddings.shape[0]
        num_docs = doc_embeddings.shape[0] // num_queries
        doc_embeddings = doc_embeddings.repeat(query_embeddings.shape[0], 1, 1)
        doc_scoring_mask = doc_scoring_mask.repeat(query_embeddings.shape[0], 1)
        scores = self.scoring_function.score(
            query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
        )
        scores = self.format_scores(num_queries, num_docs, scores)
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, labels)
        return loss

    def compute_in_batch_hinge_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_queries = query_embeddings.shape[0]
        num_docs = doc_embeddings.shape[0] // num_queries
        doc_embeddings = doc_embeddings.repeat(query_embeddings.shape[0], 1, 1)
        doc_scoring_mask = doc_scoring_mask.repeat(query_embeddings.shape[0], 1)
        scores = self.scoring_function.score(
            query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
        )
        scores = self.format_scores(num_queries, num_docs, scores)
        labels = torch.zeros(scores.shape, device=scores.device)
        labels[:, 0] = 1
        abs_scores = scores.abs()
        loss = (1 - abs_scores) * labels + abs_scores * (1 - labels)
        return loss.mean()


class MarginMSE(LossFunction):

    def __init__(
        self,
        margin: float | Literal["labels", "doc_scores"] = 1.0,
        in_batch_loss: Literal["hinge", "cross-entropy"] | None = None,
    ):
        super().__init__(in_batch_loss=in_batch_loss)
        self.margin = margin

    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scores = self.scoring_function.score(
            query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
        )
        scores = scores.view(query_embeddings.shape[0], -1)
        labels = labels.view(query_embeddings.shape[0], -1)
        # pos items are items where label is greater than other label in sample
        query_idx, pos_idx, neg_idx = torch.nonzero(
            labels[..., None] > labels[:, None], as_tuple=True
        )
        pos = scores[query_idx, pos_idx]
        neg = scores[query_idx, neg_idx]
        margin = pos - neg
        if isinstance(self.margin, float):
            target_margin = torch.tensor(self.margin, device=scores.device)
        elif self.margin == "labels":
            target_margin = labels[query_idx, pos_idx] - labels[query_idx, neg_idx]
        elif self.margin == "doc_scores":
            num_docs = doc_embeddings.shape[0] // query_embeddings.shape[0]
            target_margin = self.scoring_function.score(
                doc_embeddings[query_idx * num_docs + pos_idx],
                doc_embeddings[query_idx * num_docs + neg_idx],
                doc_scoring_mask[query_idx * num_docs + pos_idx],
                doc_scoring_mask[query_idx * num_docs + neg_idx],
            )
        else:
            raise ValueError("invalid margin type")
        loss = torch.nn.functional.mse_loss(margin, target_margin.clamp(min=0))
        return {
            "similarity loss": loss,
            **self.compute_in_batch_loss(
                query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
            ),
        }


class ConstantMarginMSE(MarginMSE):
    def __init__(
        self,
        margin: float = 1.0,
        in_batch_loss: Literal["hinge", "cross-entropy"] | None = None,
    ):
        super().__init__(margin, in_batch_loss=in_batch_loss)


class SupervisedMarginMSE(MarginMSE):
    def __init__(self, in_batch_loss: Literal["hinge", "cross-entropy"] | None = None):
        super().__init__("labels", in_batch_loss=in_batch_loss)


class DocMarginMSE(MarginMSE):
    def __init__(self, in_batch_loss: Literal["hinge", "cross-entropy"] | None = None):
        super().__init__("doc_scores", in_batch_loss)


class InBatchDocMarginMSE(DocMarginMSE):
    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_queries = query_embeddings.shape[0]
        num_docs = doc_embeddings.shape[0] // num_queries
        min_idx = torch.arange(num_queries)[:, None] * num_docs
        # max_idx = min_idx + num_docs
        # labels = (
        #     torch.arange(num_queries * num_docs)[None].greater_equal(min_idx)
        #     & torch.arange(num_queries * num_docs)[None].less(max_idx)
        # ).long()
        labels = torch.zeros(
            (num_queries, num_queries * num_docs),
            device=labels.device,
            dtype=torch.long,
        )
        labels[(torch.arange(num_queries), min_idx[:, 0])] = 1

        doc_embeddings = doc_embeddings.repeat(num_queries, 1, 1)
        doc_scoring_mask = doc_scoring_mask.repeat(num_queries, 1)

        return super().compute_loss(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
            labels,
        )


class RankNet(LossFunction):
    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scores = self.scoring_function.score(
            query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
        )
        scores = scores.view(query_embeddings.shape[0], -1)
        labels = labels.view(query_embeddings.shape[0], -1)
        greater = labels[..., None] > labels[:, None]
        diff = scores[..., None] - scores[:, None]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff[greater], torch.ones_like(diff[greater])
        )
        return {"similarity_loss": loss}


class KLDivergence(LossFunction):
    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scores = self.scoring_function.score(
            query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
        )
        scores = scores.view(query_embeddings.shape[0], -1)
        labels = labels.view(query_embeddings.shape[0], -1)
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        labels = torch.nn.functional.log_softmax(labels.to(scores), dim=-1)
        loss = torch.nn.functional.kl_div(scores, labels.to(scores), log_target=True)
        return {"similarity_loss": loss}
