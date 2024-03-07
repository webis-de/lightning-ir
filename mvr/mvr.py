from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Sequence

import torch
from lightning import LightningModule
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from transformers import PretrainedConfig, PreTrainedModel

from .data import IndexBatch, SearchBatch, TrainBatch
from .loss import LossFunction
from .tokenizer import MVRTokenizer


class MVRConfig(PretrainedConfig):
    model_type = "mvr"

    ADDED_ARGS = [
        "similarity_function",
        "aggregation_function",
        "xtr_token_retrieval_k",
        "query_expansion",
        "query_length",
        "attend_to_query_expanded_tokens",
        "doc_expansion",
        "doc_length",
        "attend_to_doc_expanded_tokens",
        "normalize",
        "add_marker_tokens",
        "embedding_dim",
        "linear_bias",
    ]

    TOKENIZER_ARGS = [
        "query_expansion",
        "query_length",
        "attend_to_query_expanded_tokens",
        "doc_expansion",
        "doc_length",
        "attend_to_doc_expanded_tokens",
        "add_marker_tokens",
    ]

    def __init__(
        self,
        similarity_function: Literal["cosine", "l2", "dot"] = "dot",
        aggregation_function: Literal["sum", "mean", "max"] = "sum",
        xtr_token_retrieval_k: int | None = None,
        query_expansion: bool = False,
        query_length: int = 32,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        doc_length: int = 512,
        attend_to_doc_expanded_tokens: bool = False,
        normalize: bool = True,
        add_marker_tokens: bool = True,
        embedding_dim: int = 128,
        linear_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.similarity_function = similarity_function
        self.aggregation_function = aggregation_function
        self.xtr_token_retrieval_k = xtr_token_retrieval_k
        self.query_expansion = query_expansion
        self.query_length = query_length
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.doc_length = doc_length
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens
        self.normalize = normalize
        self.add_marker_tokens = add_marker_tokens
        self.embedding_dim = embedding_dim
        self.linear_bias = linear_bias

    def to_mvr_dict(self) -> Dict[str, Any]:
        return {
            arg: getattr(self, arg) for arg in self.ADDED_ARGS if hasattr(self, arg)
        }

    def to_tokenizer_dict(self) -> Dict[str, Any]:
        return {arg: getattr(self, arg) for arg in self.TOKENIZER_ARGS}

    @classmethod
    def from_other(
        cls,
        config: PretrainedConfig,
        **kwargs,
    ) -> "MVRConfig":
        return cls.from_dict({**config.to_dict(), **kwargs})


def ceil_div(a: int, b: int) -> int:
    return -(a // -b)


class ScoringFunction:
    MASK_VALUE = -10000

    def __init__(
        self,
        config: MVRConfig,
    ) -> None:
        self.config = config
        if self.config.similarity_function == "cosine":
            self.similarity_function = self.cosine_similarity
        elif self.config.similarity_function == "l2":
            self.similarity_function = self.l2_similarity
        elif self.config.similarity_function == "dot":
            self.similarity_function = self.dot_similarity
        else:
            raise ValueError(
                f"Unknown similarity function {self.config.similarity_function}"
            )
        self.aggregation_function = self.config.aggregation_function

    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        to_cpu = query_embeddings.is_cpu or doc_embeddings.is_cpu
        if torch.cuda.is_available():
            query_embeddings = query_embeddings.cuda()
            doc_embeddings = doc_embeddings.cuda()

        similarity = self.similarity_function(query_embeddings, doc_embeddings)
        if to_cpu:
            similarity = similarity.cpu()
        return similarity

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)

    def l2_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -1 * torch.cdist(x, y).squeeze(-2)

    def dot_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.transpose(-1, -2)).squeeze(-2)

    @staticmethod
    def aggregate(
        scores: torch.Tensor,
        mask: torch.Tensor,
        aggregation_function: Literal["max", "sum", "mean"],
    ) -> torch.Tensor:
        scores[~mask] = 0
        if aggregation_function == "max":
            return scores.max(-1).values
        if aggregation_function == "sum":
            return scores.sum(-1)
        if aggregation_function == "mean":
            num_non_masked = mask.sum(-1)
            return scores.sum(-1) / num_non_masked
        raise ValueError(f"Unknown aggregation {aggregation_function}")

    def _parse_num_docs(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        num_docs: int | Sequence[int] | None,
    ) -> torch.Tensor:
        batch_size = query_embeddings.shape[0]
        if isinstance(num_docs, int):
            num_docs = [num_docs] * batch_size
        if isinstance(num_docs, list):
            if sum(num_docs) != doc_embeddings.shape[0] or len(num_docs) != batch_size:
                raise ValueError("Num docs does not match doc embeddings")
        if num_docs is None:
            if doc_embeddings.shape[0] % batch_size != 0:
                raise ValueError(
                    "Docs are not evenly distributed in batch, but no num_docs provided"
                )
            num_docs = [doc_embeddings.shape[0] // batch_size] * batch_size
        return torch.tensor(num_docs, device=query_embeddings.device)

    def query_scoring_mask(
        self,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scoring_mask(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            query_expansion=self.config.query_expansion,
        )

    def doc_scoring_mask(
        self,
        doc_input_ids: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scoring_mask(
            doc_input_ids, doc_attention_mask, self.config.doc_expansion
        )

    def scoring_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        query_expansion: bool,
    ) -> torch.Tensor:
        if input_ids is None:
            if attention_mask is None:
                return torch.ones(1, 1, 1, dtype=torch.bool)
            else:
                shape = attention_mask.shape
                device = attention_mask.device
        else:
            shape = input_ids.shape
            device = input_ids.device
        scoring_mask = attention_mask
        if query_expansion or scoring_mask is None:
            scoring_mask = torch.ones(shape, dtype=torch.bool, device=device)
        scoring_mask = scoring_mask.bool()
        return scoring_mask

    def scoring_masks(
        self,
        query_input_ids: torch.Tensor | None = None,
        doc_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_scoring_mask = self.query_scoring_mask(
            query_input_ids, query_attention_mask
        )
        doc_scoring_mask = self.doc_scoring_mask(doc_input_ids, doc_attention_mask)
        return query_scoring_mask, doc_scoring_mask

    def score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: int | List[int] | None = None,
    ) -> torch.Tensor:
        num_docs_t = self._parse_num_docs(query_embeddings, doc_embeddings, num_docs)

        exp_query_embeddings = query_embeddings.repeat_interleave(
            num_docs_t, dim=0
        ).unsqueeze(2)
        exp_doc_embeddings = doc_embeddings.unsqueeze(1)
        exp_query_scoring_mask = (
            query_scoring_mask.bool().repeat_interleave(num_docs_t, dim=0).unsqueeze(2)
        )
        exp_doc_scoring_mask = doc_scoring_mask.bool().unsqueeze(1)
        mask = exp_query_scoring_mask & exp_doc_scoring_mask

        similarity = self.compute_similarity(
            exp_query_embeddings, exp_doc_embeddings, mask
        )
        scores = self.aggregate(similarity, mask, "max")
        scores = self.aggregate(scores, mask.any(-1), self.aggregation_function)
        return scores


class MVRModel(PreTrainedModel):
    def __init__(self, config: MVRConfig, encoder: PreTrainedModel):
        super().__init__(config)
        self.config: MVRConfig
        self.encoder = encoder
        self.linear = torch.nn.Linear(
            self.config.hidden_size,
            self.config.embedding_dim,
            bias=self.config.linear_bias,
        )
        self.config.similarity_function
        self.scoring_function = ScoringFunction(config)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        doc_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
        query_token_type_ids: torch.Tensor | None = None,
        doc_token_type_ids: torch.Tensor | None = None,
        num_docs: List[int] | int | None = None,
    ) -> torch.Tensor:
        query_embeddings = self.encode(
            query_input_ids, query_attention_mask, query_token_type_ids
        )
        doc_embeddings = self.encode(
            doc_input_ids, doc_attention_mask, doc_token_type_ids
        )
        query_scoring_mask, doc_scoring_mask = self.scoring_masks(
            query_input_ids, doc_input_ids, query_attention_mask, doc_attention_mask
        )
        scores = self.score(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
            num_docs,
        )
        return scores

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.encoder.forward(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state
        embedding = self.linear(embedding)
        if self.config.normalize:
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(input_ids, attention_mask, token_type_ids)

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(input_ids, attention_mask, token_type_ids)

    def scoring_masks(
        self,
        query_input_ids: torch.Tensor | None = None,
        doc_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        doc_attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.scoring_function.scoring_masks(
            query_input_ids, doc_input_ids, query_attention_mask, doc_attention_mask
        )

    def score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_scoring_mask: torch.Tensor,
        doc_scoring_mask: torch.Tensor,
        num_docs: List[int] | int | None = None,
    ) -> torch.Tensor:
        if (
            query_scoring_mask.dtype != torch.bool
            or doc_scoring_mask.dtype != torch.bool
        ):
            raise ValueError("Scoring masks must be boolean")
        scores = self.scoring_function.score(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask=query_scoring_mask,
            doc_scoring_mask=doc_scoring_mask,
            num_docs=num_docs,
        )
        return scores


class MVRModule(LightningModule):
    def __init__(self, model: MVRModel, loss_function: LossFunction | None = None):
        super().__init__()
        self.model: MVRModel = model
        self.encoder: PreTrainedModel = model.encoder
        self.encoder.embeddings.position_embeddings.requires_grad_(False)
        self.config = self.model.config
        self.loss_function: LossFunction | None = loss_function
        self.tokenizer: MVRTokenizer = MVRTokenizer.from_pretrained(
            self.config.name_or_path, **self.config.to_tokenizer_dict()
        )
        if (
            self.config.add_marker_tokens
            and len(self.tokenizer) != self.config.vocab_size
        ):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer), 8)
        keys = MVRConfig().to_mvr_dict().keys()
        if any(not hasattr(self.config, key) for key in keys):
            raise ValueError(f"Model is missing MVR config attributes {keys}")

        self.validation_step_outputs = []

    def forward(self, batch: TrainBatch) -> torch.Tensor:
        num_docs = [len(ids) for ids in batch.doc_ids]
        scores = self.model.forward(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
            batch.query_encoding.token_type_ids,
            batch.doc_encoding.token_type_ids,
            num_docs,
        )
        return scores

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        if self.loss_function is None:
            raise ValueError("Loss function is not set")
        query_embeddings = self.model.encode_queries(**batch.query_encoding)
        doc_embeddings = self.model.encode_docs(**batch.doc_encoding)
        query_scoring_mask, doc_scoring_mask = self.model.scoring_masks(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids,
            batch.query_encoding.attention_mask,
            batch.doc_encoding.attention_mask,
        )
        scores = self.model.score(
            query_embeddings, doc_embeddings, query_scoring_mask, doc_scoring_mask
        )
        scores = scores.view(query_embeddings.shape[0], -1)
        targets = batch.targets.view_as(scores)
        loss = self.loss_function.compute_loss(scores, targets)
        self.log("similarity loss", loss)
        ib_loss = None
        if self.loss_function.in_batch_loss is not None:
            ib_loss = self.compute_ib_loss(batch, query_embeddings, doc_embeddings)
            self.log("ib loss", ib_loss)
        loss = loss + ib_loss if ib_loss is not None else loss
        self.log("loss", loss, prog_bar=True)
        return loss

    def compute_ib_loss(
        self,
        batch: TrainBatch,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_function is None:
            raise ValueError("Loss function is not set")
        num_queries = query_embeddings.shape[0]
        num_docs = len(batch.doc_ids[0])
        doc_embeddings = doc_embeddings.repeat(query_embeddings.shape[0], 1, 1)
        if batch.doc_encoding.attention_mask is not None:
            doc_attention_mask = batch.doc_encoding.attention_mask.repeat(
                query_embeddings.shape[0], 1
            )
        query_scoring_mask, doc_scoring_mask = self.model.scoring_masks(
            batch.query_encoding.input_ids,
            batch.doc_encoding.input_ids.repeat(query_embeddings.shape[0], 1),
            batch.query_encoding.attention_mask,
            doc_attention_mask,
        )
        ib_scores = self.model.score(
            query_embeddings,
            doc_embeddings,
            query_scoring_mask,
            doc_scoring_mask,
        )
        ib_scores = ib_scores.view(num_queries, num_docs * num_queries)
        min_idx = torch.arange(num_queries)[:, None] * num_docs
        max_idx = min_idx + num_docs
        pos_mask = torch.arange(num_queries * num_docs)[None].greater_equal(
            min_idx
        ) & torch.arange(num_queries * num_docs)[None].less(max_idx)
        pos_scores = ib_scores[pos_mask].view(-1, 1)
        neg_scores = (
            ib_scores[~pos_mask].view(num_queries, -1).repeat_interleave(num_docs, 0)
        )
        ib_scores = torch.cat((pos_scores, neg_scores), dim=1)
        ib_loss = self.loss_function.compute_in_batch_loss(ib_scores)
        return ib_loss

    def validation_step(
        self,
        batch: TrainBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        scores = self.forward(batch)
        scores = scores.view(batch.query_encoding.input_ids.shape[0], -1)
        depth = scores.shape[-1]
        relevances = batch.relevances
        assert relevances is not None
        scores = torch.nn.functional.pad(
            scores, (0, relevances.shape[-1] - scores.shape[-1])
        )
        dataset_name = ""
        first_stage = ""
        try:
            dataset_path = Path(
                self.trainer.datamodule.inference_datasets[dataloader_idx]
            )
            dataset_name = dataset_path.stem + "-"
            first_stage = dataset_path.parent.name + "-"
        except RuntimeError:
            pass

        metrics = (
            RetrievalNormalizedDCG(top_k=10, aggregation=lambda x, dim: x),
            RetrievalMRR(top_k=depth, aggregation=lambda x, dim: x),
        )
        for metric_name, metric in zip(("ndcg@10", "mrr@ranking"), (metrics)):
            value = metric(
                scores,
                relevances.clamp(0, 1) if "mrr" in metric_name else relevances,
                torch.arange(scores.shape[0])[:, None].expand_as(scores),
            )
            self.validation_step_outputs.append(
                (f"{first_stage}{dataset_name}{metric_name}", value)
            )

    def on_validation_epoch_end(self) -> None:
        aggregated = defaultdict(list)
        for key, value in self.validation_step_outputs:
            aggregated[key].extend(value)

        self.validation_step_outputs.clear()

        for key, value in aggregated.items():
            stacked = torch.stack(value)
            stacked[torch.isnan(stacked)] = 0
            self.log(key, stacked.mean(), sync_dist=True)

    def predict_step(self, batch: IndexBatch | SearchBatch, *args, **kwargs) -> Any:
        if isinstance(batch, IndexBatch):
            return self.model.encode_docs(**batch.doc_encoding)
        if isinstance(batch, SearchBatch):
            return self.model.encode_queries(**batch.query_encoding)
        raise ValueError(f"Unknown batch type {type(batch)}")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if not self.trainer.training or self.trainer.global_rank != 0:
                return
            step = self.trainer.global_step
            self.config.save_step = step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
