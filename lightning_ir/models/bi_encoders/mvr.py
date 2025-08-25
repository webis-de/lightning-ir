"""Configuration, model, and tokenizer for MVR (Multi-View Representation) type models. Originally proposed in
`Multi-View Document Representation Learning for Open-Domain Dense Retrieval \
<https://aclanthology.org/2022.acl-long.414/>`_.
"""

from typing import Literal

import torch
from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding

from lightning_ir.bi_encoder.bi_encoder_model import BiEncoderEmbedding

from ...bi_encoder import BiEncoderTokenizer, MultiVectorBiEncoderConfig, MultiVectorBiEncoderModel


class MvrConfig(MultiVectorBiEncoderConfig):
    """Configuration class for a MVR model."""

    model_type = "mvr"
    """Model type for a MVR model."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalize: bool = False,
        add_marker_tokens: bool = False,
        embedding_dim: int | None = None,
        projection: Literal["linear", "linear_no_bias"] | None = "linear",
        num_viewer_tokens: int | None = 8,
        **kwargs,
    ):
        """A MVR model encodes queries and document separately. It uses a single vector to represent the query and
        multiple vectors to represent the document. The document representation is obtained from n viewer tokens ([VIE])
        prepended to the document. During training, a contrastive loss pushes the viewer token representations away
        from one another, such that they represent different "views" of the document. Only the maximum similarity
        between the query vector and the viewer token vectors is used to compute the relevance score.

        Args:
            query_length (int): Maximum query length. Defaults to 32.
            doc_length (int): Maximum document length. Defaults to 512.
            similarity_function (Literal['cosine', 'dot']): Similarity function to compute scores between query and
                document embeddings. Defaults to "dot".
            normalize (bool): Whether to normalize query and document embeddings. Defaults to False.
            add_marker_tokens (bool): Whether to prepend extra marker tokens [Q] / [D] to queries / documents.
                Defaults to False.
            embedding_dim (int | None): Dimension of the final embeddings. If None, it will be set to the hidden size
                of the backbone model. Defaults to None.
            projection (Literal["linear", "linear_no_bias"] | None): Type of projection layer to apply on the pooled
                embeddings. If None, no projection is applied. Defaults to "linear".
            num_viewer_tokens (int | None): Number of viewer tokens to prepend to the document. Defaults to 8.
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            normalize=normalize,
            add_marker_tokens=add_marker_tokens,
            embedding_dim=embedding_dim,
            projection=projection,
            **kwargs,
        )
        self.num_viewer_tokens = num_viewer_tokens


class MvrModel(MultiVectorBiEncoderModel):
    """MVR model for multi-view representation learning."""

    config_class = MvrConfig
    """Configuration class for MVR models."""

    def __init__(self, config: MvrConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        """Initialize a MVR model.

        Args:
            config (MvrConfig): Configuration for the MVR model.
        """
        if self.config.projection is None:
            self.projection: torch.nn.Module = torch.nn.Identity()
        else:
            if self.config.embedding_dim is None:
                raise ValueError("Unable to determine embedding dimension.")
            self.projection = torch.nn.Linear(
                self.config.hidden_size,
                self.config.embedding_dim,
                bias="no_bias" not in self.config.projection,
            )

    def scoring_mask(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> torch.Tensor:
        """Computes a scoring mask for batched tokenized text sequences which is used in the scoring function to mask
        out vectors during scoring.

        Args:
            encoding (BatchEncoding): Tokenizer encodings for the text sequence.
            input_type (Literal["query", "doc"]): Type of input, either "query" or "doc".
        Returns:
            torch.Tensor: Scoring mask.
        """
        if input_type == "query":
            return torch.ones(encoding.input_ids.shape[0], 1, dtype=torch.bool, device=encoding.input_ids.device)
        elif input_type == "doc":
            return torch.ones(
                encoding.input_ids.shape[0],
                self.config.num_viewer_tokens,
                dtype=torch.bool,
                device=encoding.input_ids.device,
            )
        else:
            raise ValueError(f"Invalid input type: {input_type}")

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        """Encodes a batched tokenized text sequences and returns the embeddings and scoring mask.

        Args:
            encoding (BatchEncoding): Tokenizer encodings for the text sequence.
            input_type (Literal["query", "doc"]): Type of input, either "query" or "doc".
        Returns:
            BiEncoderEmbedding: Embeddings and scoring mask.
        """
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.projection(embeddings)
        if input_type == "query":
            embeddings = self.pooling(embeddings, None, "first")
        elif input_type == "doc":
            embeddings = embeddings[:, 1 : self.config.num_viewer_tokens + 1]
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self.scoring_mask(encoding, input_type)
        return BiEncoderEmbedding(embeddings, scoring_mask, encoding)


class MvrTokenizer(BiEncoderTokenizer):
    config_class = MvrConfig

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        add_marker_tokens: bool = False,
        num_viewer_tokens: int = 8,
        **kwargs,
    ):
        super().__init__(
            *args,
            query_length=query_length,
            doc_length=doc_length,
            add_marker_tokens=add_marker_tokens,
            num_viewer_tokens=num_viewer_tokens,
            **kwargs,
        )
        self.num_viewer_tokens = num_viewer_tokens
        if num_viewer_tokens is not None:
            viewer_tokens = [f"[VIE{idx}]" for idx in range(num_viewer_tokens)]
            self.add_tokens(viewer_tokens, special_tokens=True)
            special_tokens = [
                ("[CLS]", self.cls_token_id),
                ("[SEP]", self.sep_token_id),
            ] + [
                (viewer_tokens[viewer_token_id], self.viewer_token_id(viewer_token_id))
                for viewer_token_id in range(num_viewer_tokens)
            ]
            viewer_tokens_string = " ".join(viewer_tokens)
            if self.doc_token_id is not None:
                prefix = f"[CLS] {self.DOC_TOKEN}"
                special_tokens.append((self.DOC_TOKEN, self.doc_token_id))
            else:
                prefix = "[CLS]"
            self.doc_post_processor = TemplateProcessing(
                single=f"{prefix} {viewer_tokens_string} $0 [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=special_tokens,
            )

    def viewer_token_id(self, viewer_token_id: int) -> int | None:
        """The token id of the query token if marker tokens are added.

        :return: Token id of the query token
        :rtype: int | None
        """
        if f"[VIE{viewer_token_id}]" in self.added_tokens_encoder:
            return self.added_tokens_encoder[f"[VIE{viewer_token_id}]"]
        return None
