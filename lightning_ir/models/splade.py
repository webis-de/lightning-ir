from typing import Literal, Sequence

import torch
from transformers import BatchEncoding

from ..bi_encoder import BiEncoderConfig, BiEncoderModel, BiEncoderEmbedding
from ..modeling_utils.mlm_head import MODEL_TYPE_TO_OUTPUT_EMBEDDINGS


class SpladeConfig(BiEncoderConfig):
    model_type = "splade"

    def __init__(
        self,
        query_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "max",
        doc_pooling_strategy: Literal["first", "mean", "max", "sum"] | None = "max",
        projection: Literal["linear", "linear_no_bias", "mlm"] | None = "mlm",
        sparsification: Literal["relu", "relu_log"] | None = "relu_log",
        embedding_dim: int = 30522,
        **kwargs,
    ) -> None:
        kwargs["query_expansion"] = False
        kwargs["attend_to_query_expanded_tokens"] = False
        kwargs["query_mask_scoring_tokens"] = None
        kwargs["doc_expansion"] = False
        kwargs["attend_to_doc_expanded_tokens"] = False
        kwargs["doc_mask_scoring_tokens"] = None
        kwargs["query_aggregation_function"] = "sum"
        kwargs["normalize"] = False
        kwargs["add_marker_tokens"] = False
        super().__init__(
            query_pooling_strategy=query_pooling_strategy,
            doc_pooling_strategy=doc_pooling_strategy,
            embedding_dim=embedding_dim,
            projection=projection,
            sparsification=sparsification,
            **kwargs,
        )


class SpladeModel(BiEncoderModel):
    config_class = SpladeConfig

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        expansion = False  # getattr(self.config, f"{input_type}_expansion")
        pooling_strategy = "max"  # getattr(self.config, f"{input_type}_pooling_strategy")
        projection = self.projection if self.config.tie_projection else getattr(self, f"{input_type}_projection")
        mask_scoring_input_ids = getattr(self, f"{input_type}_mask_scoring_input_ids")

        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = projection(embeddings)
        embeddings = self._sparsification(embeddings, self.config.sparsification)
        embeddings = self._pooling(embeddings, encoding["attention_mask"], pooling_strategy)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        scoring_mask = self.scoring_mask(encoding, expansion, pooling_strategy, mask_scoring_input_ids)
        return BiEncoderEmbedding(embeddings, scoring_mask, encoding)

    def set_output_embeddings(self, new_embeddings: torch.nn.Module) -> None:
        if self.config.projection == "mlm":
            raise NotImplementedError("Setting output embeddings is not supported for models with MLM projection.")
            module_names = MODEL_TYPE_TO_OUTPUT_EMBEDDINGS[self.config.backbone_model_type or self.config.model_type]
            module = self
            for module_name in module_names.split(".")[:-1]:
                module = getattr(module, module_name)
            setattr(module, module_names.split(".")[-1], new_embeddings)
            setattr(module, "bias", new_embeddings.bias)

    def get_output_embeddings(self) -> torch.nn.Module | None:
        """Returns the output embeddings of the model for tieing the input and output embeddings. Returns None if no
        MLM head is used for projection.

        :return: Output embeddings of the model
        :rtype: torch.nn.Module | None
        """

        class _TiedOutputEmbeddingsContainer(torch.nn.Module):
            """This is a hack to tie the output embeddings of multiple layers. HF only supports tieing the output of a
            single layer at the moment. This hack will lead to errors if the input embedding dimensionality is changed,
            e.g., a new token is added to the vocabulary."""

            def __init__(self, output_embeddings: Sequence[torch.nn.Linear]):
                super().__init__()
                self.output_embeddings = output_embeddings
                self.weight = output_embeddings[0].weight

            def __setattr__(self, name: str, value: torch.Tensor | torch.nn.Module) -> None:
                if name == "weight":
                    for output_embedding in self.output_embeddings:
                        output_embedding.weight = value
                super().__setattr__(name, value)

        if self.config.projection == "mlm":
            module_names = MODEL_TYPE_TO_OUTPUT_EMBEDDINGS[self.config.backbone_model_type or self.config.model_type]
            if self.config.tie_projection:
                output = self.projection
                for module_name in module_names.split("."):
                    output = getattr(output, module_name)
                return output
            else:
                query_output = self.query_projection
                doc_output = self.doc_projection
                for module_name in module_names.split("."):
                    query_output = getattr(query_output, module_name)
                    doc_output = getattr(doc_output, module_name)
                container = _TiedOutputEmbeddingsContainer([query_output, doc_output])
                return container
        return None
