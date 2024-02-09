import warnings
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import torch
from lightning import LightningModule
from tokenizers.processors import TemplateProcessing
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMRR
from transformers import (
    BatchEncoding,
    BertTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
)

from tide.data import Batch
from tide.loss import LossFunction


class ScoringFunction:
    MASK_VALUE = -10000

    def __init__(
        self,
        similarity_function: Literal["cosine", "l2"],
        query_aggregation_function: Literal["sum", "mean", "max"],
        doc_aggregation_function: Literal["sum", "mean", "max"],
    ) -> None:
        if similarity_function == "cosine":
            self.similarity_function = self.cosine_similarity
        elif similarity_function == "l2":
            self.similarity_function = self.l2_similarity
        else:
            raise ValueError(f"Unknown similarity function {similarity_function}")
        self.query_aggregation_function = query_aggregation_function
        self.doc_aggregation_function = doc_aggregation_function

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.transpose(-1, -2)).squeeze(-2)

    def l2_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # TODO untested
        return 1 - torch.dist(x, y)

    def aggregate(
        self,
        similarity: torch.Tensor,
        aggregate_func: str,
    ) -> torch.Tensor:
        if aggregate_func == "max":
            return similarity.max(-1)[0]
        mask = similarity == self.MASK_VALUE
        similarity[mask] = 0
        if aggregate_func == "sum":
            return similarity.sum(-1)
        if aggregate_func == "mean":
            num_non_masked = mask.logical_not().sum(-1)
            return similarity.sum(-1) / num_non_masked
        raise ValueError(f"Unknown aggregation {aggregate_func}")

    def reformat_docs(
        self,
        doc_embeddings: torch.Tensor,
        doc_attention_mask: torch.Tensor | None,
        num_docs: int | List[int] | None,
        batch_size: int,
        embedding_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int]:
        if isinstance(num_docs, list):
            if sum(num_docs) != doc_embeddings.shape[0] or len(num_docs) != batch_size:
                raise ValueError("Num docs does not match doc embeddings")
            if len(set(num_docs)) == 1:
                num_docs = num_docs[0]
        if num_docs is None:
            if doc_embeddings.shape[0] % batch_size != 0:
                raise ValueError(
                    "Docs are not evenly distributed in batch, but no num_docs provided"
                )
            num_docs = doc_embeddings.shape[0] // batch_size
        similarity_mask = None
        if not isinstance(num_docs, int):
            doc_embeddings = torch.nn.utils.rnn.pad_sequence(
                list(torch.split(doc_embeddings, num_docs)),
                batch_first=True,
                padding_value=self.MASK_VALUE,
            )
            similarity_mask = doc_embeddings.eq(self.MASK_VALUE).all(-1).all(-1)
            if doc_attention_mask is not None:
                doc_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    list(torch.split(doc_attention_mask, num_docs)),
                    batch_first=True,
                    padding_value=0,
                )
            num_docs = max(num_docs)
        doc_embeddings = doc_embeddings.view(batch_size, num_docs, 1, -1, embedding_dim)
        if doc_attention_mask is not None:
            doc_attention_mask = doc_attention_mask.view(batch_size, num_docs, 1, -1)
        return doc_embeddings, doc_attention_mask, similarity_mask, num_docs

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        query_attention_mask: torch.Tensor | None,
        doc_embeddings: torch.Tensor,
        doc_attention_mask: torch.Tensor | None,
        num_docs: List[int] | int | None,
    ) -> torch.Tensor:
        batch_size, query_len, embedding_dim = query_embeddings.shape

        masked_tokens = doc_embeddings.eq(self.MASK_VALUE).all(-1)
        if masked_tokens.any():
            if doc_attention_mask is None:
                doc_attention_mask = torch.ones_like(masked_tokens)
            doc_attention_mask = (doc_attention_mask.bool() & ~masked_tokens).long()

        doc_embeddings, doc_attention_mask, similarity_mask, num_docs = (
            self.reformat_docs(
                doc_embeddings, doc_attention_mask, num_docs, batch_size, embedding_dim
            )
        )

        query_embeddings = query_embeddings.view(
            batch_size, 1, query_len, 1, embedding_dim
        )
        if query_attention_mask is not None:
            query_attention_mask = query_attention_mask.view(
                batch_size, 1, query_len, 1
            )

        similarity = self.similarity_function(query_embeddings, doc_embeddings)
        if query_attention_mask is not None:
            query_mask = ~query_attention_mask.bool().expand_as(similarity)
            similarity[query_mask] = self.MASK_VALUE
        if doc_attention_mask is not None:
            doc_mask = ~doc_attention_mask.bool().expand_as(similarity)
            similarity[doc_mask] = self.MASK_VALUE

        similarity = self.aggregate(similarity, self.doc_aggregation_function)
        similarity = self.aggregate(similarity, self.query_aggregation_function)
        if similarity_mask is not None:
            similarity[similarity_mask] = self.MASK_VALUE
        return similarity


class MVRConfig(PretrainedConfig):

    model_type = "mvr"

    ADDED_ARGS = [
        "similarity_function",
        "query_aggregation_function",
        "doc_aggregation_function",
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
        similarity_function: Literal["cosine", "l2"] = "cosine",
        query_aggregation_function: Literal["sum", "mean", "max"] = "sum",
        doc_aggregation_function: Literal["sum", "mean", "max"] = "max",
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
        self.query_aggregation_function = query_aggregation_function
        self.doc_aggregation_function = doc_aggregation_function
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


class MVRMixin:

    config: MVRConfig
    encoder: PreTrainedModel
    linear: torch.nn.Linear

    @property
    def scoring_function(self) -> ScoringFunction:
        return ScoringFunction(
            self.config.similarity_function,
            self.config.query_aggregation_function,
            self.config.doc_aggregation_function,
        )

    def search(
        self, query_embeddings: torch.Tensor, query_attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._encode(input_ids, attention_mask, token_type_ids)

    def _encode(
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
        return self._encode(input_ids, attention_mask, token_type_ids)

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._encode(input_ids, attention_mask, token_type_ids)

    def score(
        self,
        query_embeddings: torch.Tensor,
        query_attention_mask: torch.Tensor | None,
        doc_embeddings: torch.Tensor,
        doc_attention_mask: torch.Tensor | None,
        num_docs: List[int] | None,
    ) -> torch.Tensor:
        query_attention_mask = (
            torch.ones(query_embeddings.shape[:2], dtype=torch.long)
            if self.config.query_expansion
            else query_attention_mask
        )
        doc_attention_mask = (
            torch.ones(doc_embeddings.shape[:2], dtype=torch.long)
            if self.config.doc_expansion
            else doc_attention_mask
        )
        score = self.scoring_function(
            query_embeddings,
            query_attention_mask,
            doc_embeddings,
            doc_attention_mask,
            num_docs,
        )
        return score


class MVRModel(MVRMixin, PreTrainedModel, ABC):
    pass


class MVRTokenizer(BertTokenizerFast):

    def __init__(
        self,
        vocab_file: str | Path | None = None,
        tokenizer_file: str | Path | None = None,
        do_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        query_token: str = "[QUE]",
        doc_token: str = "[DOC]",
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        query_expansion: bool = False,
        query_length: int = 32,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        doc_length: int = 512,
        attend_to_doc_expanded_tokens: bool = False,
        add_marker_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file,
            do_lower_case,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            tokenize_chinese_chars,
            strip_accents,
            query_expansion=query_expansion,
            query_length=query_length,
            attend_to_query_expanded_tokens=attend_to_query_expanded_tokens,
            doc_expansion=doc_expansion,
            doc_length=doc_length,
            attend_to_doc_expanded_tokens=attend_to_doc_expanded_tokens,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
        self.query_expansion = query_expansion
        self.query_length = query_length
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.doc_length = doc_length
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens

        self._query_token = query_token
        self._doc_token = doc_token

        self.query_post_processor = None
        self.doc_post_processor = None
        if add_marker_tokens:
            self.add_tokens([query_token, doc_token], special_tokens=True)
            self.query_post_processor = TemplateProcessing(
                single="[CLS] [QUE] $0 [SEP]",
                pair="[CLS] [QUE] $A [SEP] [DOC] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    ("[QUE]", self.query_token_id),
                    ("[DOC]", self.doc_token_id),
                ],
            )
            self.doc_post_processor = TemplateProcessing(
                single="[CLS] [DOC] $0 [SEP]",
                pair="[CLS] [SEP] $A [SEP] [DOC] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    ("[QUE]", self.query_token_id),
                    ("[DOC]", self.doc_token_id),
                ],
            )

    def save_pretrained(
        self,
        save_directory: str | Path,
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        return super().save_pretrained(
            save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs
        )

    @property
    def query_token(self) -> str:
        return self._query_token

    @property
    def doc_token(self) -> str:
        return self._doc_token

    @property
    def query_token_id(self) -> int | None:
        if self.query_token in self.added_tokens_encoder:
            return self.added_tokens_encoder[self.query_token]
        return None

    @property
    def doc_token_id(self) -> int | None:
        if self.doc_token in self.added_tokens_encoder:
            return self.added_tokens_encoder[self.doc_token]
        return None

    def __call__(self, *args, internal: bool = False, **kwargs) -> BatchEncoding:
        if not internal:
            warnings.warn(
                "MVRTokenizer is directly called. Use encode_queries or encode_docs "
                "if marker_tokens should be added and query/doc expansion applied."
            )
        return super().__call__(*args, **kwargs)

    def _encode(
        self,
        text: str | List[str],
        *args,
        post_processor: TemplateProcessing | None = None,
        **kwargs,
    ) -> BatchEncoding:
        orig_post_processor = self._tokenizer.post_processor
        if post_processor is not None:
            self._tokenizer.post_processor = post_processor
        encoding = self(text, *args, internal=True, **kwargs)
        self._tokenizer.post_processor = orig_post_processor
        return encoding

    def _expand(
        self, encoding: BatchEncoding, attend_to_expanded_tokens: bool
    ) -> BatchEncoding:
        input_ids = encoding["input_ids"]
        input_ids[input_ids == self.pad_token_id] = self.mask_token_id
        encoding["input_ids"] = input_ids
        if attend_to_expanded_tokens:
            encoding["attention_mask"] = None
        return encoding

    def encode_queries(
        self, queries: List[str] | str, *args, **kwargs
    ) -> BatchEncoding:
        if self.query_expansion:
            kwargs["max_length"] = self.query_length
            kwargs["padding"] = "max_length"
        encoding = self._encode(
            queries, post_processor=self.query_post_processor, *args, **kwargs
        )
        if self.query_expansion:
            self._expand(encoding, self.attend_to_query_expanded_tokens)
        return encoding

    def encode_docs(self, docs: List[str] | str, *args, **kwargs) -> BatchEncoding:
        if self.doc_expansion:
            kwargs["max_length"] = self.doc_length
            kwargs["padding"] = "max_length"
        encoding = self._encode(
            docs, post_processor=self.doc_post_processor, *args, **kwargs
        )
        if self.doc_expansion:
            self._expand(encoding, self.attend_to_doc_expanded_tokens)
        return encoding


class MVRModule(LightningModule):
    def __init__(self, model: MVRModel, loss_function: LossFunction | None) -> None:
        super().__init__()
        self.model: MVRModel = model
        self.encoder: PreTrainedModel = model.encoder
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

    def forward(self, batch: Batch):
        query_embeddings = self.model.encode_queries(**batch.query_encoding)
        doc_embeddings = self.model.encode_docs(**batch.doc_encoding)
        num_docs = [len(docs) for docs in batch.doc_ids]
        scores = self.model.score(
            query_embeddings,
            batch.query_encoding.attention_mask,
            doc_embeddings,
            batch.doc_encoding.attention_mask,
            num_docs,
        )
        return scores

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        scores = self.forward(batch)
        targets = batch.targets.view_as(scores)
        if self.loss_function is None:
            raise ValueError("Loss function is not set")
        loss = self.loss_function(scores, targets)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        scores = self.forward(batch)
        depth = scores.shape[-1]
        relevance = batch.relevance
        assert relevance is not None
        scores = torch.nn.functional.pad(
            scores, (0, relevance.shape[-1] - scores.shape[-1])
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
                relevance.clamp(0, 1) if "mrr" in metric_name else relevance,
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
