import array
from pathlib import Path

import torch

from ...bi_encoder import BiEncoderConfig, BiEncoderOutput
from ...data import IndexBatch
from ..base import IndexConfig, Indexer


class TorchDenseIndexer(Indexer):
    def __init__(
        self,
        index_dir: Path,
        index_config: "TorchDenseIndexConfig",
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, bi_encoder_config, verbose)
        self.embeddings = array.array("f")

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        doc_embeddings = output.doc_embeddings
        if doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")

        doc_lengths = doc_embeddings.scoring_mask.sum(dim=1)
        embeddings = doc_embeddings.embeddings[doc_embeddings.scoring_mask]
        num_docs = len(index_batch.doc_ids)
        self.doc_ids.extend(index_batch.doc_ids)
        self.doc_lengths.extend(doc_lengths.cpu().tolist())
        self.num_embeddings += embeddings.shape[0]
        self.num_docs += num_docs
        self.embeddings.extend(embeddings.cpu().view(-1).float().tolist())

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def save(self) -> None:
        super().save()
        index = torch.frombuffer(self.embeddings, dtype=torch.float32).view(self.num_embeddings, -1)
        torch.save(index, self.index_dir / "index.pt")


class TorchDenseIndexConfig(IndexConfig):
    indexer_class = TorchDenseIndexer
