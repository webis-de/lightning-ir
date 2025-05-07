import array
from pathlib import Path

import torch

from ...bi_encoder import BiEncoderModule, BiEncoderOutput
from ...data import IndexBatch
from ...models import SpladeConfig
from ..base import IndexConfig, Indexer


class TorchSparseIndexer(Indexer):
    def __init__(
        self,
        index_dir: Path,
        index_config: "TorchSparseIndexConfig",
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, module, verbose)
        self.crow_indices = array.array("L")
        self.crow_indices.append(0)
        self.col_indices = array.array("L")
        self.values = array.array("f")

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
        num_docs = len(index_batch.doc_ids)
        self.doc_ids.extend(index_batch.doc_ids)

        token_idcs, dim_idcs = torch.nonzero(embeddings, as_tuple=True)
        crow_indices = token_idcs.bincount().cumsum(0) + self.crow_indices[-1]
        values = embeddings[token_idcs, dim_idcs]
        self.crow_indices.extend(crow_indices.cpu().tolist())
        self.col_indices.extend(dim_idcs.cpu().tolist())
        self.values.extend(values.cpu().tolist())

        self.doc_lengths.extend(doc_lengths.int().cpu().tolist())
        self.num_embeddings += embeddings.shape[0]
        self.num_docs += num_docs

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def save(self) -> None:
        super().save()
        index = torch.sparse_csr_tensor(
            torch.frombuffer(self.crow_indices, dtype=torch.int64),
            torch.frombuffer(self.col_indices, dtype=torch.int64),
            torch.frombuffer(self.values, dtype=torch.float32),
            torch.Size([self.num_embeddings, self.module.config.embedding_dim]),
        )
        torch.save(index, self.index_dir / "index.pt")


class TorchSparseIndexConfig(IndexConfig):
    indexer_class = TorchSparseIndexer
    SUPPORTED_MODELS = {SpladeConfig.model_type}
