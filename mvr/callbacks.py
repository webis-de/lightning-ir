from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, Sequence

import faiss
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback

from mvr.datamodule import RUN_HEADER, MVRDataModule
from mvr.mvr import MVRModule


class PredictionWriter(BasePredictionWriter):
    def __init__(self, overwrite: bool = False) -> None:
        super().__init__("batch")
        self.overwrite = overwrite

    def get_run_path(
        self, trainer: Trainer, pl_module: LightningModule, dataset_idx: int
    ) -> Path:
        if not hasattr(trainer, "ckpt_path") or trainer.ckpt_path is None:
            raise ValueError("ckpt_path is not set")
        if not hasattr(trainer, "datamodule") or trainer.datamodule is None:
            raise ValueError("datamodule is not set")
        ckpt_path = Path(trainer.ckpt_path)
        datamodule: MVRDataModule = trainer.datamodule
        if (
            datamodule.inference_datasets is None
            or datamodule.inference_run_config is None
        ):
            raise ValueError("inference datasets are not set")
        inference_run_path = Path(datamodule.inference_datasets[dataset_idx])
        dataset_id = inference_run_path.stem
        first_stage = inference_run_path.parent.name
        filename = (
            f"{first_stage}_{datamodule.inference_run_config.depth}_{dataset_id}.run"
        )
        run_file_path = ckpt_path.parent.parent / "runs" / filename
        return run_file_path

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: MVRModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        run_file_path = self.get_run_path(trainer, pl_module, dataloader_idx)
        doc_ids = batch["doc_ids"]
        query_ids = batch["query_id"]
        scores = [float(logit.item()) for logits in prediction for logit in logits]
        query_ids = [
            query_id
            for batch_idx, query_id in enumerate(query_ids)
            for _ in range(len(doc_ids[batch_idx]))
        ]
        doc_ids = [doc_id for doc_ids in doc_ids for doc_id in doc_ids]
        run_df = pd.DataFrame(
            zip(query_ids, doc_ids, scores), columns=["query", "docid", "score"]
        )
        run_df = run_df.sort_values(["query", "score"], ascending=[True, False])
        run_df["rank"] = (
            run_df.groupby("query")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        run_df["q0"] = 0
        run_df["system"] = pl_module.config.name_or_path
        run_df = run_df[RUN_HEADER]
        run_file_path.parent.mkdir(exist_ok=True)
        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"
        run_df.to_csv(run_file_path, header=False, index=False, sep="\t", mode=mode)


def get_num_elements(
    file_path: Path, dtype: str, shape: Sequence[int] | None = None
) -> int:
    file_size = file_path.stat().st_size
    sum_shape = 1
    if shape is not None:
        if all(dim != -1 for dim in shape):
            raise ValueError("shape must contain exactly one -1")
        sum_shape = sum(shape) + 1
    num_elements = file_size // (sum_shape * np.dtype(dtype).itemsize)
    return num_elements


class IndexConfig(NamedTuple):
    index_path: Path
    num_train_tokens: int
    save_doc_lengths: bool = False
    num_centroids: int = 65536
    num_subquantizers: int = 16
    code_size: int = 4


class IndexCallback(Callback):
    def __init__(self, config: IndexConfig) -> None:
        super().__init__()
        self.config = config
        self.index: faiss.Index
        self._doc_ids = []
        self._doc_lengths = []
        self._num_embeddings = 0
        self._num_docs = 0
        self._train_embeddings: np.ndarray | None
        self._index_path: Path

    def setup(self, trainer: Trainer, pl_module: MVRModule, stage: str) -> None:
        if stage != "predict":
            raise ValueError("IndexingCallback can only be used in predict stage")
        if pl_module.config.similarity_function == "l2":
            coarse_quantizer = faiss.IndexFlatL2(pl_module.config.embedding_dim)
        elif pl_module.config.similarity_function in ("cosine", "dot"):
            coarse_quantizer = faiss.IndexFlatIP(pl_module.config.embedding_dim)
        else:
            raise ValueError(
                f"similarity_function {pl_module.config.similarity_function} unknown"
            )
        self.index = faiss.IndexIVFPQ(
            coarse_quantizer,
            pl_module.config.embedding_dim,
            self.config.num_centroids,
            self.config.num_subquantizers,
            self.config.code_size,
        )
        self.index.make_direct_map()
        if pl_module.config.similarity_function == "cosine":
            self.index = faiss.IndexPreTransform(
                faiss.NormalizationTransform, self.index
            )

        self._train_embeddings = np.empty(
            (self.config.num_train_tokens, pl_module.config.embedding_dim),
            dtype=np.float32,
        )

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("predict_dataloaders are not set")
        if len(dataloaders) != 1:
            raise ValueError("IndexingCallback can only be used with one dataloader")
        dataset = dataloaders[0].dataset
        dataset_id = dataset.dataset.dataset_id()
        self._index_path = self.config.index_path / dataset_id
        self._index_path.mkdir(parents=True, exist_ok=True)

    def _grab_train_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self._train_embeddings is not None:
            # save training embeddings until num_train_tokens is reached
            # if num_train_tokens overflows, save the remaining embeddings
            start = self._num_embeddings
            end = start + embeddings.shape[0]
            if end > self.config.num_train_tokens:
                end = self.config.num_train_tokens
            length = end - start
            self._train_embeddings[start:end] = embeddings[:length]
            self._num_embeddings += length
            embeddings = embeddings[length:]
        return embeddings

    def _train(self):
        if (
            self._train_embeddings is not None
            and self._num_embeddings >= self.config.num_train_tokens
        ):
            self.index.train(self._train_embeddings)
            self.index.add(self._train_embeddings)
            self._train_embeddings = None

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: MVRModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        outputs = pl_module.all_gather(outputs)
        encoded_doc_ids = torch.ByteTensor(
            list(bytes(doc_id.zfill(20), "utf8") for doc_id in batch.doc_ids)
        )
        doc_ids = pl_module.all_gather(encoded_doc_ids)
        if trainer.is_global_zero:
            outputs = outputs.view(-1, *outputs.shape[-2:])
            masked = (outputs == 0).all(-1)
            if self.config.save_doc_lengths:
                self._doc_lengths.append(
                    masked.logical_not().sum(-1).cpu().numpy().astype(np.uint16)
                )
            outputs = outputs.view(-1, pl_module.config.embedding_dim)
            embeddings = outputs[~masked.view(-1)].cpu().numpy().astype(np.float32)

            embeddings = self._grab_train_embeddings(embeddings)
            self._train()
            self._num_embeddings += embeddings.shape[0]
            self._num_docs += doc_ids.shape[0]
            if embeddings.shape[0]:
                self.index.add(embeddings)
            self._doc_ids.append(doc_ids.view(-1, 20).cpu().numpy())

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.index.is_trained:
            raise ValueError("index is not trained")
        if self._num_embeddings != self.index.ntotal:
            raise ValueError("number of embeddings does not match index.ntotal")
        doc_ids_fp = np.memmap(
            self._index_path / "doc_ids.npy",
            dtype="uint8",
            mode="w+",
            shape=(self._num_docs, 20),
        )
        doc_lengths_fp = None
        if self.config.save_doc_lengths:
            doc_lengths_fp = np.memmap(
                self._index_path / "doc_lengths.npy",
                dtype="uint16",
                mode="w+",
                shape=(self._num_docs,),
            )
        num_tokens = 0
        iterator = zip(self._doc_ids, self._doc_lengths or [None] * len(self._doc_ids))
        for doc_ids, doc_lengths in iterator:
            start = num_tokens
            end = start + doc_ids.shape[0]
            num_tokens += doc_ids.shape[0]
            doc_ids_fp[start:end] = doc_ids
            if doc_lengths is not None and doc_lengths_fp is not None:
                doc_lengths_fp[start:end] = doc_lengths
        doc_ids_fp.flush()
        if doc_lengths_fp is not None:
            doc_lengths_fp.flush()

        faiss.write_index(self.index, str(self._index_path / "index.faiss"))


class SearchConfig(NamedTuple):
    index_path: Path
    k: int
    imputation_strategy: Literal["min", "gather"]


class SearchCallback(Callback):
    def __init__(self, config: SearchConfig) -> None:
        super().__init__()
        self.config = config
        self.index: faiss.Index
        self.num_docs: int
        self.doc_ids: np.memmap
        self.cumulative_doc_lengths: np.ndarray | None
        self.cumulative_doc_lengths: np.ndarray | None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "predict":
            raise ValueError("SearchCallback can only be used in predict stage")
        if (
            self.config.imputation_strategy == "gather"
            and not (self.config.index_path / "doc_lengths.npy").exists()
        ):
            raise ValueError("index needs to contain doc lengths for gather imputation")

        self.index = faiss.read_index(str(self.config.index_path / "index.faiss"))
        self.doc_ids = np.memmap(
            self.config.index_path / "doc_ids.npy",
            dtype="S20",
            mode="r",
        )
        self.num_docs = self.doc_ids.shape[0]
        self.cumulative_doc_lengths = None
        if (self.config.index_path / "doc_lengths.npy").exists():
            doc_lengths = np.memmap(
                self.config.index_path / "doc_lengths.npy",
                dtype="uint16",
                mode="r",
            )
            self.doc_lengths = np.empty_like(doc_lengths)
            self.doc_lengths[:] = doc_lengths[:]
            if (
                self.doc_lengths.shape[0] != self.num_docs
                or self.doc_lengths.sum() != self.index.ntotal
            ):
                raise ValueError("doc_lengths do not match index")
            self.cumulative_doc_lengths = np.cumsum(doc_lengths)

    def _gather_imputation(self, token_ids: np.ndarray) -> torch.Tensor:
        if self.cumulative_doc_lengths is None:
            raise ValueError("doc_lengths are not set")
        doc_ids = np.searchsorted(self.cumulative_doc_lengths, token_ids, side="right")
        # TODO gather doc_embeddings using self.index.reconstruct_n
        self.index.reconstruct_batch(token_ids)

    def min_imputation(self, token_ids: np.ndarray) -> torch.Tensor:
        pass

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: MVRModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # self.index.reconstruct_from_offset could be useful
        outputs = pl_module.all_gather(outputs)
        if trainer.is_global_zero:
            outputs = outputs.view(-1, *outputs.shape[-2:])
            masked = (outputs == 0).all(-1)
            query_length = masked.logical_not().sum(-1).cpu().numpy()
            query_tokens = (
                outputs[masked.logical_not()].cpu().numpy().astype(np.float32)
            )
            # TODO token_embeddings necessary or just directly use score?!
            score, token_ids, token_embeddings = self.index.search_and_reconstruct(
                query_tokens, self.config.k
            )
            if self.config.imputation_strategy == "gather":
                doc_token_embeddings, doc_lengths = self._gather_imputation(token_ids)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
