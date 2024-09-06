import warnings
from dataclasses import dataclass
from functools import wraps
from string import punctuation
from typing import Callable, Iterable, Literal, Sequence, Tuple, overload

import torch
from transformers import BatchEncoding
from transformers.activations import ACT2FN

from ..base import LightningIRModel, LightningIROutput
from ..base.model import _batch_encoding
from . import BiEncoderConfig


class MLMHead(torch.nn.Module):
    def __init__(self, config: BiEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


@dataclass
class BiEncoderEmbedding:
    embeddings: torch.Tensor
    scoring_mask: torch.Tensor

    @overload
    def to(self, device: torch.device, /) -> "BiEncoderEmbedding": ...

    @overload
    def to(self, other: "BiEncoderEmbedding", /) -> "BiEncoderEmbedding": ...

    def to(self, device) -> "BiEncoderEmbedding":
        if isinstance(device, BiEncoderEmbedding):
            device = device.device
            self.embeddings.to()
        self.embeddings = self.embeddings.to(device)
        self.scoring_mask = self.scoring_mask.to(device)
        return self

    @property
    def device(self) -> torch.device:
        if self.embeddings.device != self.scoring_mask.device:
            raise ValueError("Embeddings and scoring_mask must be on the same device")
        return self.embeddings.device

    def items(self) -> Iterable[Tuple[str, torch.Tensor]]:
        for field in self.__dataclass_fields__:
            yield field, getattr(self, field)


@dataclass
class BiEncoderOutput(LightningIROutput):
    query_embeddings: BiEncoderEmbedding | None = None
    doc_embeddings: BiEncoderEmbedding | None = None


class BiEncoderModel(LightningIRModel):

    _tied_weights_keys = ["projection.decoder.bias", "projection.decoder.weight", "encoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    config_class = BiEncoderConfig

    def __init__(self, config: BiEncoderConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: BiEncoderConfig
        self.scoring_function = ScoringFunction(self.config)
        self.projection: torch.nn.Linear | MLMHead | None = None
        if self.config.projection is not None:
            if "linear" in self.config.projection:
                self.projection = torch.nn.Linear(
                    self.config.hidden_size,
                    self.config.embedding_dim,
                    bias="no_bias" not in self.config.projection,
                )
            elif self.config.projection == "mlm":
                self.projection = MLMHead(config)
            else:
                raise ValueError(f"Unknown projection {self.config.projection}")
        else:
            if self.config.embedding_dim != self.config.hidden_size:
                warnings.warn(
                    "No projection is used but embedding_dim != hidden_size. "
                    "The output embeddings will not have embedding_size dimensions."
                )

        self.query_mask_scoring_input_ids: torch.Tensor | None = None
        self.doc_mask_scoring_input_ids: torch.Tensor | None = None
        self.add_mask_scoring_input_ids()

    @classmethod
    def _load_pretrained_model(
        cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
    ):
        if model.config.projection == "mlm":
            has_base_model_prefix = any(s.startswith(model.base_model_prefix) for s in state_dict.keys())
            prefix = model.base_model_prefix + "." if has_base_model_prefix else ""
            for key in list(state_dict.keys()):
                if key.startswith("cls"):
                    new_key = prefix + key.replace("cls.predictions", "projection").replace(".transform", "")
                    state_dict[new_key] = state_dict.pop(key)
                    loaded_keys[loaded_keys.index(key)] = new_key
        return super()._load_pretrained_model(
            model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
        )

    def get_output_embeddings(self) -> torch.nn.Module | None:
        if isinstance(self.projection, MLMHead):
            return self.projection.decoder
        return None

    def add_mask_scoring_input_ids(self) -> None:
        for sequence in ("query", "doc"):
            mask_scoring_tokens = getattr(self.config, f"{sequence}_mask_scoring_tokens")
            if mask_scoring_tokens is None:
                continue
            if mask_scoring_tokens == "punctuation":
                mask_scoring_tokens = list(punctuation)
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
            except OSError:
                raise ValueError("Can't use token scoring masking if the checkpoint does not have a tokenizer.")
            mask_scoring_input_ids = []
            for token in mask_scoring_tokens:
                if token not in tokenizer.vocab:
                    raise ValueError(f"Token {token} not in tokenizer vocab")
                mask_scoring_input_ids.append(tokenizer.vocab[token])
            setattr(
                self,
                f"{sequence}_mask_scoring_input_ids",
                torch.tensor(mask_scoring_input_ids, dtype=torch.long),
            )

    def forward(
        self,
        query_encoding: BatchEncoding | None,
        doc_encoding: BatchEncoding | None,
        num_docs: Sequence[int] | int | None = None,
    ) -> BiEncoderOutput:
        query_embeddings = None
        if query_encoding is not None:
            query_embeddings = self.encode_query(query_encoding)
        doc_embeddings = None
        if doc_encoding is not None:
            doc_embeddings = self.encode_doc(doc_encoding)
        scores = None
        if doc_embeddings is not None and query_embeddings is not None:
            scores = self.score(query_embeddings, doc_embeddings, num_docs)
        return BiEncoderOutput(scores=scores, query_embeddings=query_embeddings, doc_embeddings=doc_embeddings)

    def encode_query(self, encoding: BatchEncoding) -> BiEncoderEmbedding:
        return self._encode(
            encoding,
            expansion=self.config.query_expansion,
            pooling_strategy=self.config.query_pooling_strategy,
            mask_scoring_input_ids=self.query_mask_scoring_input_ids,
        )

    def encode_doc(self, encoding: BatchEncoding) -> BiEncoderEmbedding:
        return self._encode(
            encoding,
            expansion=self.config.doc_expansion,
            pooling_strategy=self.config.doc_pooling_strategy,
            mask_scoring_input_ids=self.doc_mask_scoring_input_ids,
        )

    @_batch_encoding
    def _encode(
        self,
        encoding: BatchEncoding,
        expansion: bool = False,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
        mask_scoring_input_ids: torch.Tensor | None = None,
    ) -> BiEncoderEmbedding:
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        embeddings = self._sparsification(embeddings, self.config.sparsification)
        embeddings = self._pooling(embeddings, encoding["attention_mask"], pooling_strategy)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self._scoring_mask(
            encoding["input_ids"],
            encoding["attention_mask"],
            expansion,
            pooling_strategy,
            mask_scoring_input_ids,
        )
        return BiEncoderEmbedding(embeddings, scoring_mask)

    def query_scoring_mask(self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None) -> torch.Tensor:
        return self._scoring_mask(
            input_ids,
            attention_mask,
            expansion=self.config.query_expansion,
            pooling_strategy=self.config.query_pooling_strategy,
            mask_scoring_input_ids=self.config.query_mask_scoring_input_ids,
        )

    def doc_scoring_mask(self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None) -> torch.Tensor:
        return self._scoring_mask(
            input_ids,
            attention_mask,
            expansion=self.config.query_expansion,
            pooling_strategy=self.config.doc_pooling_strategy,
            mask_scoring_input_ids=self.config.doc_mask_scoring_input_ids,
        )

    def _scoring_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        expansion: bool,
        pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
        mask_scoring_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            shape = input_ids.shape
            device = input_ids.device
        elif attention_mask is not None:
            shape = attention_mask.shape
            device = attention_mask.device
        else:
            raise ValueError("Pass either input_ids or attention_mask")
        if pooling_strategy is not None:
            return torch.ones((shape[0], 1), dtype=torch.bool, device=device)
        scoring_mask = attention_mask
        if expansion or scoring_mask is None:
            scoring_mask = torch.ones(shape, dtype=torch.bool, device=device)
        scoring_mask = scoring_mask.bool()
        if mask_scoring_input_ids is not None and input_ids is not None:
            ignore_mask = input_ids[..., None].eq(mask_scoring_input_ids.to(device)).any(-1)
            scoring_mask = scoring_mask & ~ignore_mask
        return scoring_mask

    def score(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        scores = self.scoring_function.score(query_embeddings, doc_embeddings, num_docs=num_docs)
        return scores


def _batch_scoring(
    similarity_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    BATCH_SIZE = 1024

    @wraps(similarity_function)
    def batch_similarity_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= BATCH_SIZE:
            return similarity_function(x, y)
        out = torch.zeros(x.shape[0], x.shape[1], y.shape[2], device=x.device, dtype=x.dtype)
        for i in range(0, x.shape[0], BATCH_SIZE):
            out[i : i + BATCH_SIZE] = similarity_function(x[i : i + BATCH_SIZE], y[i : i + BATCH_SIZE])
        return out

    return batch_similarity_function


class ScoringFunction(torch.nn.Module):
    def __init__(self, config: BiEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.similarity_function == "cosine":
            self.similarity_function = self.cosine_similarity
        elif self.config.similarity_function == "l2":
            self.similarity_function = self.l2_similarity
        elif self.config.similarity_function == "dot":
            self.similarity_function = self.dot_similarity
        else:
            raise ValueError(f"Unknown similarity function {self.config.similarity_function}")
        self.query_aggregation_function = self.config.query_aggregation_function

    def compute_similarity(
        self, query_embeddings: BiEncoderEmbedding, doc_embeddings: BiEncoderEmbedding
    ) -> torch.Tensor:
        # if torch.cuda.is_available():
        #     # bfloat16 similarity yields weird values with gpu, so we use fp16 instead
        #     # TODO investigate why, all values are a factor of 1/4
        #     query_tensor = query_tensor.cuda().half()
        #     doc_tensor = doc_tensor.cuda().half()

        # TODO compute similarity only for non-masked values
        similarity = self.similarity_function(query_embeddings.embeddings, doc_embeddings.embeddings)
        return similarity

    @staticmethod
    @_batch_scoring
    def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)

    @staticmethod
    @_batch_scoring
    def l2_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -1 * torch.cdist(x, y).squeeze(-2)

    @staticmethod
    @_batch_scoring
    def dot_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.transpose(-1, -2)).squeeze(-2)

    def parse_num_docs(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: int | Sequence[int] | None,
    ) -> torch.Tensor:
        batch_size = query_embeddings.embeddings.shape[0]
        if isinstance(num_docs, int):
            num_docs = [num_docs] * batch_size
        if isinstance(num_docs, list):
            if sum(num_docs) != doc_embeddings.embeddings.shape[0] or len(num_docs) != batch_size:
                raise ValueError("Num docs does not match doc embeddings")
        if num_docs is None:
            if doc_embeddings.embeddings.shape[0] % batch_size != 0:
                raise ValueError("Docs are not evenly distributed in _batch, but no num_docs provided")
            num_docs = [doc_embeddings.embeddings.shape[0] // batch_size] * batch_size
        return torch.tensor(num_docs, device=query_embeddings.embeddings.device)

    def expand_query_embeddings(
        self,
        embeddings: BiEncoderEmbedding,
        num_docs: torch.Tensor,
    ) -> BiEncoderEmbedding:
        return BiEncoderEmbedding(
            embeddings.embeddings.repeat_interleave(num_docs, dim=0).unsqueeze(2),
            embeddings.scoring_mask.repeat_interleave(num_docs, dim=0).unsqueeze(2),
        )

    def expand_doc_embeddings(
        self,
        embeddings: BiEncoderEmbedding,
        num_docs: torch.Tensor,
    ) -> BiEncoderEmbedding:
        return BiEncoderEmbedding(embeddings.embeddings.unsqueeze(1), embeddings.scoring_mask.unsqueeze(1))

    def aggregate(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor | None,
        query_aggregation_function: Literal["max", "sum", "mean", "harmonic_mean"] | None,
        dim: int,
    ) -> torch.Tensor:
        if query_aggregation_function is None:
            return scores
        if query_aggregation_function == "max":
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            return scores.max(dim, keepdim=True).values
        if query_aggregation_function == "sum":
            if mask is not None:
                scores = scores.masked_fill(~mask, 0)
            return scores.sum(dim, keepdim=True)
        if mask is None:
            shape = list(scores.shape)
            shape[dim] = 1
            num_non_masked = torch.full(shape, scores.shape[dim], device=scores.device)
        else:
            num_non_masked = mask.sum(dim, keepdim=True)
        if query_aggregation_function == "mean":
            return torch.where(num_non_masked == 0, 0, scores.sum(dim, keepdim=True) / num_non_masked)
        if query_aggregation_function == "harmonic_mean":
            return torch.where(
                num_non_masked == 0,
                0,
                num_non_masked / (1 / scores).sum(dim, keepdim=True),
            )
        raise ValueError(f"Unknown aggregation {query_aggregation_function}")

    def score(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        num_docs_t = self.parse_num_docs(query_embeddings, doc_embeddings, num_docs)
        query_embeddings = self.expand_query_embeddings(query_embeddings, num_docs_t)
        doc_embeddings = self.expand_doc_embeddings(doc_embeddings, num_docs_t)
        similarity = self.compute_similarity(query_embeddings, doc_embeddings)
        scores = self.aggregate(similarity, doc_embeddings.scoring_mask, "max", -1)
        scores = self.aggregate(scores, query_embeddings.scoring_mask, self.query_aggregation_function, -2)
        return scores[..., 0, 0]
