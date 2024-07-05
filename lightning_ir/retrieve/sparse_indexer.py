import array
from pathlib import Path
import torch

from ..bi_encoder import BiEncoderConfig, BiEncoderOutput
from ..data import IndexBatch
from .indexer import IndexConfig, Indexer


class SparseIndexConfig(IndexConfig):
    pass


class SparseIndexer(Indexer):

    def __init__(
        self,
        index_dir: Path,
        index_config: IndexConfig,
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        super().__init__(index_dir, index_config, bi_encoder_config, verbose)
        self.token_idcs = array.array("I")
        self.dim_idcs = array.array("I")
        self.values = array.array("f")

    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        doc_embeddings = output.doc_embeddings
        if doc_embeddings is None:
            raise ValueError("Expected doc_embeddings in BiEncoderOutput")

        doc_lengths = doc_embeddings.scoring_mask.sum(dim=1)
        embeddings = doc_embeddings.embeddings[doc_embeddings.scoring_mask]
        num_docs = len(index_batch.doc_ids)
        self.doc_ids.extend(index_batch.doc_ids)

        token_idcs, dim_idcs = torch.nonzero(embeddings, as_tuple=True)
        values = embeddings[token_idcs, dim_idcs]
        self.token_idcs.extend((token_idcs + self.num_embeddings).cpu().tolist())
        self.dim_idcs.extend(dim_idcs.cpu().tolist())
        self.values.extend(values.cpu().tolist())

        self.doc_lengths.extend(doc_lengths.cpu().tolist())
        self.num_embeddings += embeddings.shape[0]
        self.num_docs += num_docs

    def to_gpu(self) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def save(self) -> None:
        super().save()
        index = torch.sparse_coo_tensor(
            torch.stack([torch.tensor(self.token_idcs), torch.tensor(self.dim_idcs)]),
            torch.tensor(self.values),
            torch.Size([self.num_embeddings, self.bi_encoder_config.embedding_dim]),
        )
        torch.save(index, self.index_dir / "index.pt")
