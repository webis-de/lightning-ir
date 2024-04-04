from abc import ABC, abstractmethod
from typing import Dict, Literal, Tuple

import torch

from .mvr import ScoringFunction


class LossFunction(ABC):
    def __init__(self):
        self.scoring_function: ScoringFunction

    def set_scoring_function(self, scoring_function: ScoringFunction):
        self.scoring_function = scoring_function

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


class SimilarityLossFunction(LossFunction):
    pass


class InBatchLossFunction(LossFunction):

    def __init__(
        self,
        pos_sampling_technique: Literal["all", "first"] = "all",
        neg_sampling_technique: Literal["all", "first"] = "all",
    ):
        super().__init__()
        self.pos_sampling_technique = pos_sampling_technique
        self.neg_sampling_technique = neg_sampling_technique

    def format_doc_embeddings(
        self,
        num_queries: int,
        doc_embeddings: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_total_docs, seq_len, emb_dim = doc_embeddings.shape
        num_docs = num_total_docs // num_queries
        doc_embeddings = doc_embeddings.repeat(num_queries, 1, 1).view(
            num_queries,
            num_docs * num_queries,
            seq_len,
            emb_dim,
        )
        doc_scoring_mask = doc_scoring_mask.repeat(num_queries, 1).view(
            num_queries, num_docs * num_queries, seq_len
        )
        min_idx = torch.arange(num_queries)[:, None] * num_docs
        max_idx = min_idx + num_docs
        if self.pos_sampling_technique == "all":
            pos_mask = torch.arange(num_queries * num_docs)[None].greater_equal(
                min_idx
            ) & torch.arange(num_queries * num_docs)[None].less(max_idx)
        elif self.pos_sampling_technique == "first":
            pos_mask = torch.arange(num_queries * num_docs)[None].eq(min_idx)
        else:
            raise ValueError("invalid pos sampling technique")
        if self.neg_sampling_technique == "all":
            neg_mask = torch.arange(num_queries * num_docs)[None].less(
                min_idx
            ) | torch.arange(num_queries * num_docs)[None].greater_equal(max_idx)
        elif self.neg_sampling_technique == "first":
            neg_mask = torch.arange(num_queries * num_docs)[None, None].eq(min_idx).any(
                1
            ) & torch.arange(num_queries * num_docs)[None].ne(min_idx)
        else:
            raise ValueError("invalid neg sampling technique")
        doc_embeddings = torch.cat(
            [
                doc_embeddings[pos_mask].view(num_queries, -1, seq_len, emb_dim),
                doc_embeddings[neg_mask].view(num_queries, -1, seq_len, emb_dim),
            ],
            dim=1,
        ).view(-1, seq_len, emb_dim)
        doc_scoring_mask = torch.cat(
            [
                doc_scoring_mask[pos_mask].view(num_queries, -1, seq_len),
                doc_scoring_mask[neg_mask].view(num_queries, -1, seq_len),
            ],
            dim=1,
        ).view(-1, seq_len)
        return doc_embeddings, doc_scoring_mask

    def get_in_batch_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
    ) -> torch.Tensor:
        doc_embeddings, doc_scoring_mask = self.format_doc_embeddings(
            query_embeddings.shape[0], doc_embeddings, doc_scoring_mask
        )
        scores = self.scoring_function.score(
            query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
        )
        return scores


class InBatchCrossEntropy(InBatchLossFunction):

    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scores = self.get_in_batch_scores(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
        )
        scores = scores.view(query_embeddings.shape[0], -1)
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, labels)
        return {self.__class__.__name__: loss}


class InBatchHinge(InBatchLossFunction):

    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scores = self.get_in_batch_scores(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
        )
        labels = torch.zeros(scores.shape, device=scores.device)
        labels[:, 0] = 1
        abs_scores = scores.abs()
        loss = (1 - abs_scores) * labels + abs_scores * (1 - labels)
        return {self.__class__.__name__: loss.mean()}


class MarginMSE(SimilarityLossFunction):

    def __init__(self, margin: float | Literal["labels", "doc_scores"] = 1.0):
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
        return {self.__class__.__name__: loss}


class ConstantMarginMSE(MarginMSE):
    def __init__(self, margin: float = 1.0):
        super().__init__(margin)


class SupervisedMarginMSE(MarginMSE):
    def __init__(self):
        super().__init__("labels")


class DocMarginMSE(MarginMSE):
    def __init__(self):
        super().__init__("doc_scores")


class RankNet(SimilarityLossFunction):
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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            margin, torch.ones_like(margin)
        )
        return {self.__class__.__name__: loss}


class KLDivergence(SimilarityLossFunction):
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
        return {self.__class__.__name__: loss}


class QueryLevelCrossEntropy(SimilarityLossFunction):
    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.view(query_embeddings.shape[0], -1)
        query_idx, pos_idx, neg_idx = torch.nonzero(
            labels[..., None] > labels[:, None], as_tuple=True
        )
        num_docs_t = self.scoring_function.parse_num_docs(
            query_embeddings, doc_embeddings, None
        )
        query_embeddings, query_scoring_mask = (
            self.scoring_function.expand_query_embeddings(
                query_embeddings, query_scoring_mask, num_docs_t
            )
        )
        doc_embeddings, doc_scoring_mask = self.scoring_function.expand_doc_embeddings(
            doc_embeddings, doc_scoring_mask, num_docs_t
        )
        mask = query_scoring_mask & doc_scoring_mask
        similarity = self.scoring_function.compute_similarity(
            query_embeddings, doc_embeddings, mask, num_docs_t
        )
        similarity = self.scoring_function.aggregate(similarity, mask, "max")
        similarity = similarity.view(*labels.shape, -1)
        diff = similarity[query_idx, pos_idx] - similarity[query_idx, neg_idx]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff, torch.ones_like(diff)
        )
        return {self.__class__.__name__: loss}
