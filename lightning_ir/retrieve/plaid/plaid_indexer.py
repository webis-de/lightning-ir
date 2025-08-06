import warnings
from array import array
from pathlib import Path

import torch

from ...bi_encoder import BiEncoderModule, BiEncoderOutput
from ...data import IndexBatch
from ...models import ColConfig
from ..base import IndexConfig, Indexer
from .residual_codec import ResidualCodec


class PlaidIndexer(Indexer):

    def __init__(
        self,
        index_dir: Path,
        index_config: "PlaidIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, module, verbose)

        self.index_config: PlaidIndexConfig

        self._train_embeddings: torch.Tensor | None = torch.full(
            (self.index_config.num_train_embeddings, self.module.config.embedding_dim),
            torch.nan,
            dtype=torch.float32,
        )
        self.residual_codec: ResidualCodec | None = None
        self.codes = array("l")
        self.residuals = array("B")

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        doc_embeddings = output.doc_embeddings
        if doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")

        if doc_embeddings.scoring_mask is None:
            doc_lengths = torch.ones(
                doc_embeddings.embeddings.shape[0], device=doc_embeddings.device, dtype=torch.int32
            )
            embeddings = doc_embeddings.embeddings[:, 0]
        else:
            doc_lengths = doc_embeddings.scoring_mask.sum(dim=1)
            embeddings = doc_embeddings.embeddings[doc_embeddings.scoring_mask]
        doc_ids = index_batch.doc_ids
        embeddings = self.process_embeddings(embeddings)

        if embeddings.shape[0]:
            if self.residual_codec is None:
                raise ValueError("Residual codec not trained")
            codes, residuals = self.residual_codec.compress(embeddings)
            self.codes.extend(codes.numpy(force=True))
            self.residuals.extend(residuals.view(-1).numpy(force=True))

        self.num_embeddings += embeddings.shape[0]
        self.num_docs += len(doc_ids)

        self.doc_lengths.extend(doc_lengths.int().cpu().tolist())
        self.doc_ids.extend(doc_ids)

    def process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self._grab_train_embeddings(embeddings)
        self._train()
        return embeddings

    def _grab_train_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._train_embeddings is not None:
            # save training embeddings until num_train_embeddings is reached
            # if num_train_embeddings overflows, save the remaining embeddings
            start = self.num_embeddings
            end = min(self.index_config.num_train_embeddings, start + embeddings.shape[0])
            length = end - start
            self._train_embeddings[start:end] = embeddings[:length]
            self.num_embeddings += length
            embeddings = embeddings[length:]
        return embeddings

    def _train(self, force: bool = False) -> None:
        if self._train_embeddings is None:
            return
        if not force and self.num_embeddings < self.index_config.num_train_embeddings:
            return

        if torch.isnan(self._train_embeddings).any():
            warnings.warn("Corpus contains less tokens/documents than num_train_embeddings. Removing NaN embeddings.")
            self._train_embeddings = self._train_embeddings[~torch.isnan(self._train_embeddings).any(dim=1)]

        self.residual_codec = ResidualCodec.train(self.index_config, self._train_embeddings, self.verbose)
        codes, residuals = self.residual_codec.compress(self._train_embeddings)
        self.codes.extend(codes.numpy(force=True))
        self.residuals.extend(residuals.view(-1).numpy(force=True))

        self._train_embeddings = None

    def save(self) -> None:
        if self.residual_codec is None:
            self._train(force=True)
        if self.residual_codec is None:
            raise ValueError("No residual codec to save")
        super().save()

        codes = torch.frombuffer(self.codes, dtype=torch.long)
        residuals = torch.frombuffer(self.residuals, dtype=torch.uint8)
        torch.save(codes, self.index_dir / "codes.pt")
        torch.save(residuals, self.index_dir / "residuals.pt")
        self.residual_codec.save(self.index_dir)


class PlaidIndexConfig(IndexConfig):
    indexer_class = PlaidIndexer
    SUPPORTED_MODELS = {ColConfig.model_type}

    def __init__(
        self,
        num_centroids: int,
        num_train_embeddings: int | None = None,
        k_means_iters: int = 4,
        n_bits: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        max_points_per_centroid = 256
        self.num_centroids = num_centroids
        self.num_train_embeddings = num_train_embeddings or num_centroids * max_points_per_centroid
        self.k_means_iters = k_means_iters
        self.n_bits = n_bits
        self.seed = seed
