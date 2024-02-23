import itertools
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback

from .datamodule import RUN_HEADER, DocDataset, QueryDataset
from .indexer import IndexConfig, Indexer
from .mvr import MVRModule
from .searcher import SearchConfig, Searcher


class IndexCallback(Callback):
    def __init__(
        self,
        index_path: Path | None,
        num_train_tokens: int,
        num_centroids: int = 65536,
        num_subquantizers: int = 16,
        n_bits: int = 4,
    ) -> None:
        super().__init__()
        self.index_path = index_path
        self.num_train_tokens = num_train_tokens
        self.num_centroids = num_centroids
        self.num_subquantizers = num_subquantizers
        self.n_bits = n_bits
        self.config: IndexConfig
        self.indexer: Indexer

    def setup(self, trainer: Trainer, pl_module: MVRModule, stage: str) -> None:
        if stage != "predict":
            raise ValueError("IndexCallback can only be used in predict stage")

    def on_predict_start(self, trainer: Trainer, pl_module: MVRModule) -> None:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        if len(dataloaders) != 1:
            raise ValueError("IndexCallback can only be used with one dataloader")
        dataset = dataloaders[0].dataset
        if not isinstance(dataset, DocDataset):
            raise ValueError("Expected a DocDataset for indexing")

        index_path = self.index_path
        if index_path is None:
            if Path(pl_module.config.name_or_path).exists():
                index_dir = Path(pl_module.config.name_or_path) / "indexes"
                index_path = index_dir / dataset.docs_dataset_id
            else:
                raise ValueError(
                    "No index_path provided and model_name_or_path is not a path"
                )
        self.config = IndexConfig(
            index_path=index_path,
            num_train_tokens=self.num_train_tokens,
            num_centroids=self.num_centroids,
            num_subquantizers=self.num_subquantizers,
            n_bits=self.n_bits,
        )
        self.indexer = Indexer(self.config, pl_module.config)

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
            list(bytes(doc_id.rjust(20), "utf8") for doc_id in batch.doc_ids)
        )
        doc_ids = pl_module.all_gather(encoded_doc_ids)
        if trainer.is_global_zero:
            outputs = outputs.view(-1, *outputs.shape[-2:])

            masked = (outputs == 0).all(-1)
            doc_lengths = masked.logical_not().sum(-1).cpu().numpy().astype(np.uint16)

            outputs = outputs.view(-1, pl_module.config.embedding_dim)
            embeddings = outputs[~masked.view(-1)].cpu().numpy().astype(np.float32)

            doc_ids = doc_ids.view(-1, 20).cpu().numpy()
            self.indexer.add(embeddings, doc_ids, doc_lengths)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.indexer.save()


class SearchCallback(BasePredictionWriter):
    def __init__(
        self,
        save_dir: Path | None = None,
        index_path: Path | None = None,
        k: int = 100,
        candidate_k: int = 1000,
        imputation_strategy: Literal["min", "gather"] | None = None,
        n_probe: int = 1,
    ) -> None:
        super().__init__()
        if imputation_strategy is None:
            raise ValueError("imputation_strategy must be set")
        self.save_dir = save_dir
        self.index_path = index_path
        self.k = k
        self.candidate_k = candidate_k
        self.imputation_strategy = imputation_strategy
        self.n_probe = n_probe
        self.config: SearchConfig
        self.searcher: Searcher

    def setup(self, trainer: Trainer, pl_module: MVRModule, stage: str) -> None:
        if stage != "predict":
            raise ValueError("SearchCallback can only be used in predict stage")

    def on_predict_start(self, trainer: Trainer, pl_module: MVRModule) -> None:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        datasets = [dataloader.dataset for dataloader in dataloaders]
        if not all(isinstance(dataset, QueryDataset) for dataset in datasets):
            raise ValueError("Expected QueryDatasets for searching")
        docs_dataset_id = [dataset.docs_dataset_id for dataset in datasets]
        if len(set(docs_dataset_id)) != 1:
            raise ValueError("All QueryDatasets must have the same docs_dataset_id")
        docs_dataset_id = docs_dataset_id[0]

        index_path = self.index_path
        if index_path is None:
            if Path(pl_module.config.name_or_path).exists():
                index_dir = Path(pl_module.config.name_or_path) / "indexes"
                index_path = index_dir / docs_dataset_id
                if not index_path.exists():
                    raise ValueError(f"No index found at {index_path}")
            else:
                raise ValueError(
                    "No index_path provided and model_name_or_path is not a path"
                )
        self.config = SearchConfig(
            index_path,
            self.k,
            self.candidate_k,
            self.imputation_strategy,
            self.n_probe,
        )
        self.searcher = Searcher(self.config, pl_module.config)
        if self.save_dir is None:
            if not hasattr(trainer, "ckpt_path") or trainer.ckpt_path is None:
                raise ValueError("No save_dir provided and ckpt_path is not set")
            self.save_dir = Path(trainer.ckpt_path).parent.parent / "runs"

    def get_run_path(self, trainer: Trainer, dataset_idx: int) -> Path:
        dataloaders = trainer.predict_dataloaders
        if self.save_dir is None:
            raise ValueError("No save_dir found; call setup before using this method")
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        dataset = dataloaders[dataset_idx].dataset
        if not isinstance(dataset, QueryDataset):
            raise ValueError("Expected a QueryDataset for searching")
        dataset_id = dataset.dataset_id.replace("/", "-")
        run_file_path = self.save_dir / f"{dataset_id}.run"
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
        prediction = pl_module.all_gather(prediction)
        if not trainer.is_global_zero:
            return

        prediction = prediction.view(-1, *prediction.shape[-2:])
        masked = (prediction == 0).all(-1)
        query_lengths = masked.logical_not().sum(-1).cpu().numpy()
        query_tokens = prediction[masked.logical_not()].cpu().numpy().astype(np.float32)
        scores, doc_ids, num_docs = self.searcher.search(query_tokens, query_lengths)

        query_ids = list(
            itertools.chain.from_iterable(
                itertools.repeat(query_id, num)
                for query_id, num in zip(batch.query_ids, num_docs)
            )
        )
        run_df = pd.DataFrame(
            zip(query_ids, doc_ids, scores), columns=["query_id", "doc_id", "score"]
        )
        run_df = run_df.sort_values(["query_id", "score"], ascending=[True, False])
        run_df["rank"] = (
            run_df.groupby("query_id")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        run_df["q0"] = 0
        run_df["system"] = pl_module.config.name_or_path
        run_df = run_df[RUN_HEADER]

        run_file_path = self.get_run_path(trainer, dataloader_idx)
        run_file_path.parent.mkdir(exist_ok=True)
        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"

        run_df.to_csv(run_file_path, header=False, index=False, sep="\t", mode=mode)
