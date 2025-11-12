"""Configuration, model, and tokenizer for XTR (ContXextualized Token Retriever) type models. Originally proposed in
`Rethinking the Role of Token Retrieval in Multi-Vector Retrieval \
<https://proceedings.neurips.cc/paper_files/paper/2023/file/31d997278ee9069d6721bc194174bb4c-Paper-Conference.pdf>`_."""

from typing import Literal, Sequence

import torch

from ...bi_encoder import BiEncoderOutput

from .col import ColConfig, ColModel


class XTRConfig(ColConfig):
    """Configuration class for XTR model."""

    model_type = "xtr"
    """Model type for XTR model."""

    def __init__(
        self,
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalization: Literal["l2"] | None = "l2",
        sparsification: None | Literal["relu", "relu_log", "relu_2xlog"] = None,
        add_marker_tokens: bool = False,
        query_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        query_aggregation_function: Literal["sum", "mean", "max"] = "mean",
        doc_aggregation_function: Literal["sum", "mean", "max"] = "max",
        embedding_dim: int = 128,
        projection: Literal["linear", "linear_no_bias"] = "linear_no_bias",
        k_train: int = 128,
        use_in_batch_token_retrieval: bool = True,
        **kwargs,
    ):
        """An XTR model encodes queries and documents into sets of contextualized token embeddings. XTR introduces an
        in-batch token retrieval objective during training: for each query token, it retrieves the top-k most relevant
        document tokens from all documents in the batch, encouraging the model to focus on the most salient information.
        This mechanism allows XTR to learn fine-grained token-level interactions and improves retrieval quality by
        leveraging richer token-level matching, while maintaining scalability for large collections. At inference, XTR
        can fall back to standard ColBERT-style max-sim scoring for efficient retrieval.

        Args:
            query_length: Maximum query length in tokens. Defaults to 32.
            doc_length: Maximum document length in tokens. Defaults to 512.
            similarity_function: Similarity function for token scores. Defaults to "cosine".
            normalization: Whether to L2 normalize embeddings. Defaults to "l2".
            add_marker_tokens: Whether to add marker tokens [Q]/[D]. Defaults to False.
            query_mask_scoring_tokens: Tokens to mask during query scoring. Defaults to "punctuation".
            doc_mask_scoring_tokens: Tokens to mask during document scoring. Defaults to "punctuation".
            query_aggregation_function: How to aggregate over query tokens. Defaults to "mean".
            doc_aggregation_function: How to aggregate over document tokens. Defaults to "max".
            embedding_dim: Output embedding dimension. Defaults to 128.
            projection: Projection layer type. Defaults to "linear_no_bias".
            k_train: Number of top-k document tokens to retrieve per query token during in-batch
                training. Defaults to 128. Should be much smaller than batch_size * doc_length.
                Typical values: 32-64 (small batches), 128 (default), 256-320 (large batches).
                Note: If the batch is too small, k_train will be automatically clamped to the available number of tokens.
            use_in_batch_token_retrieval: Whether to use XTR's in-batch token retrieval during
                training. If False, falls back to standard max-sim scoring. Defaults to True.
        """
        if query_mask_scoring_tokens is not None or doc_mask_scoring_tokens is not None:
            raise NotImplementedError("Masking specific tokens is not yet implemented in XTRConfig.")
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            normalization=normalization,
            sparsification=sparsification,
            add_marker_tokens=add_marker_tokens,
            query_mask_scoring_tokens=query_mask_scoring_tokens,
            doc_mask_scoring_tokens=doc_mask_scoring_tokens,
            query_aggregation_function=query_aggregation_function,
            doc_aggregation_function=doc_aggregation_function,
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.projection = projection
        self.k_train = k_train
        self.use_in_batch_token_retrieval = use_in_batch_token_retrieval


class XTRModel(ColModel):
    """XTR bi-encoder model. See :class:`XTRConfig` for model configuration details."""

    config_class = XTRConfig
    """Configuration class for the model."""

    def score(
        self,
        output: BiEncoderOutput,
        num_docs: Sequence[int] | int | None = None,
    ) -> BiEncoderOutput:
        """Compute relevance scores between queries and documents.

        Args:
            output (BiEncoderOutput): Output containing embeddings and scoring mask.
            num_docs (Sequence[int] | int | None): Specifies how many documents are passed per query. If a sequence of
                integers, `len(num_doc)` should be equal to the number of queries and `sum(num_docs)` equal to the
                number of documents, i.e., the sequence contains one value per query specifying the number of documents
                for that query. If an integer, assumes an equal number of documents per query. If None, tries to infer
                the number of documents by dividing the number of documents by the number of queries. Defaults to None.
        Returns:
            BiEncoderOutput: Output containing relevance scores.
        """
        if self.training and self.config.use_in_batch_token_retrieval:
            return self._score_xtr_in_batch(output, num_docs)

        return super().score(output, num_docs)

    def _score_xtr_in_batch(
        self, output: BiEncoderOutput, num_docs: Sequence[int] | int | None = None
    ) -> BiEncoderOutput:
        """XTR in-batch token retrieval scoring.

        Args:
            output (BiEncoderOutput): Output containing embeddings and scoring mask.
            num_docs (Sequence[int] | int | None): Specifies how many documents are passed per query. If a sequence of
                integers, `len(num_doc)` should be equal to the number of queries and `sum(num_docs)` equal to the
                number of documents, i.e., the sequence contains one value per query specifying the number of documents
                for that query. If an integer, assumes an equal number of documents per query. If None, tries to infer
                the number of documents by dividing the number of documents by the number of queries. Defaults to None.
        Returns:
            BiEncoderOutput: Output containing relevance scores.
        """
        similarities = self.compute_similarity(output.query_embeddings, output.doc_embeddings, num_docs)

        query_mask = output.query_embeddings.scoring_mask
        doc_mask = output.doc_embeddings.scoring_mask

        num_docs_t = self._parse_num_docs(
            output.query_embeddings.embeddings.shape[0],
            output.doc_embeddings.embeddings.shape[0],
            num_docs,
            output.query_embeddings.device,
        )

        query_mask_expanded = query_mask.repeat_interleave(num_docs_t, dim=0).unsqueeze(-1)
        doc_mask_expanded = doc_mask.unsqueeze(1)

        similarities = similarities.masked_fill(~doc_mask_expanded, float("-inf"))
        similarities = similarities.masked_fill(~query_mask_expanded, float("-inf"))

        batch_size = output.query_embeddings.embeddings.shape[0]
        q_len = output.query_embeddings.embeddings.shape[1]
        doc_len = output.doc_embeddings.embeddings.shape[1]
        max_docs = torch.max(num_docs_t)

        sim_list = similarities.split(num_docs_t.tolist(), dim=0)
        sim_padded = torch.nn.utils.rnn.pad_sequence(sim_list, batch_first=True, padding_value=float("-inf"))

        valid_mask = torch.arange(max_docs, device=num_docs_t.device).unsqueeze(0) < num_docs_t.unsqueeze(1)

        sim_flat = sim_padded.view(batch_size, -1)
        k_train = min(self.config.k_train, sim_flat.size(-1))
        minimum_values = torch.topk(sim_flat, k=k_train, dim=-1).values[:, -1].unsqueeze(-1)

        sim_padded = sim_padded.view(batch_size, -1)
        sim_padded = sim_padded.masked_fill(sim_padded < minimum_values, 0.0)
        sim_padded = sim_padded.view(batch_size, max_docs, q_len, doc_len)

        scores = sim_padded.max(dim=-1).values.sum(dim=-1)
        Z = (sim_padded.max(dim=-1).values > 0).sum(dim=-1).float()
        Z = Z.clamp(min=1.0)
        scores = scores / Z

        scores = scores[valid_mask]

        output.scores = scores
        output.similarity = sim_padded[valid_mask]

        return output
