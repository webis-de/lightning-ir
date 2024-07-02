from __future__ import annotations

import itertools
from dataclasses import is_dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback, TQDMProgressBar

from ..data import IndexBatch, RankBatch, SearchBatch
from ..data.dataset import RUN_HEADER, DocDataset, QueryDataset, RunDataset
from ..retrieve import (
    FaissFlatIndexConfig,
    FaissFlatIndexer,
    FaissIVFPQIndexConfig,
    FaissIVFPQIndexer,
    SparseIndexConfig,
    SparseIndexer,
    IndexConfig,
    Indexer,
    SearchConfig,
    Searcher,
    FaissSearcher,
    SparseSearcher,
    FaissSearchConfig,
    SparseSearchConfig,
)

if TYPE_CHECKING:
    from ..base import LightningIRModule, LightningIROutput
    from ..bi_encoder import BiEncoderModule, BiEncoderOutput
    from ..cross_encoder import CrossEncoderModule, CrossEncoderOutput

T = TypeVar("T")


def format_large_number(number: float) -> str:
    suffixes = ["", "K", "M", "B", "T"]
    suffix_index = 0

    while number >= 1000 and suffix_index < len(suffixes) - 1:
        number /= 1000.0
        suffix_index += 1

    formatted_number = "{:.2f}".format(number)

    suffix = suffixes[suffix_index]
    if suffix:
        formatted_number += f" {suffix}"
    return formatted_number


class GatherMixin:
    def gather(self, pl_module: LightningIRModule, dataclass: T) -> T:
        if is_dataclass(dataclass):
            return dataclass.__class__(
                **{
                    k: self.gather(pl_module, getattr(dataclass, k))
                    for k in dataclass.__dataclass_fields__
                }
            )
        return pl_module.all_gather(dataclass)


class IndexCallback(Callback, GatherMixin):
    def __init__(
        self,
        index_dir: Path | None,
        index_config: IndexConfig,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.index_dir = index_dir
        self.index_config = index_config
        self.verbose = verbose
        self.indexer: Indexer

    def setup(self, trainer: Trainer, pl_module: BiEncoderModule, stage: str) -> None:
        if stage != "predict":
            raise ValueError("IndexCallback can only be used in predict stage")

    def on_predict_start(self, trainer: Trainer, pl_module: BiEncoderModule) -> None:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        datasets = [dataloader.dataset for dataloader in dataloaders]
        if not all(isinstance(dataset, DocDataset) for dataset in datasets):
            raise ValueError("Expected DocDatasets for indexing")

    def get_index_dir(self, pl_module: BiEncoderModule, dataset: DocDataset) -> Path:
        index_dir = self.index_dir
        if index_dir is None:
            default_index_dir = Path(pl_module.config.name_or_path)
            if default_index_dir.exists():
                index_dir = default_index_dir / "indexes"
            else:
                raise ValueError(
                    "No index_dir provided and model_name_or_path is not a path"
                )
        index_dir = index_dir / dataset.docs_dataset_id
        return index_dir

    def get_indexer(
        self, trainer: Trainer, pl_module: BiEncoderModule, dataset_idx: int
    ) -> Indexer:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        dataset = dataloaders[dataset_idx].dataset

        index_dir = self.get_index_dir(pl_module, dataset)

        if self.index_config.similarity_function is None:
            self.index_config.similarity_function = pl_module.config.similarity_function

        if isinstance(self.index_config, FaissFlatIndexConfig):
            indexer = FaissFlatIndexer(
                index_dir, self.index_config, pl_module.config, self.verbose
            )
        elif isinstance(self.index_config, FaissIVFPQIndexConfig):
            indexer = FaissIVFPQIndexer(
                index_dir, self.index_config, pl_module.config, self.verbose
            )
        elif isinstance(self.index_config, SparseIndexConfig):
            indexer = SparseIndexer(
                index_dir, self.index_config, pl_module.config, self.verbose
            )
        else:
            raise ValueError(
                f"Unsupported IndexConfig {self.index_config.__class__.__name__}"
            )
        return indexer

    def log_to_pg(self, info: Dict[str, Any], trainer: Trainer):
        pg_callback = trainer.progress_bar_callback
        if pg_callback is None or not isinstance(pg_callback, TQDMProgressBar):
            return
        pg = pg_callback.predict_progress_bar
        info = {k: format_large_number(v) for k, v in info.items()}
        if pg is not None:
            pg.set_postfix(info)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: BiEncoderModule,
        prediction: BiEncoderOutput,
        batch: IndexBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            if hasattr(self, "indexer"):
                self.indexer.save()
            self.indexer = self.get_indexer(trainer, pl_module, dataloader_idx)

        batch = self.gather(pl_module, batch)
        prediction = self.gather(pl_module, prediction)

        if not trainer.is_global_zero:
            return

        self.indexer.add(batch, prediction)
        self.log_to_pg(
            {
                "num_docs": self.indexer.num_docs,
                "num_embeddings": self.indexer.num_embeddings,
            },
            trainer,
        )

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.indexer.save()


class RankCallback(BasePredictionWriter, GatherMixin):
    def __init__(self, save_dir: Path | None = None) -> None:
        super().__init__()
        self.save_dir = save_dir

    def setup(
        self,
        trainer: Trainer,
        pl_module: BiEncoderModule | CrossEncoderModule,
        stage: str,
    ) -> None:
        if stage != "predict":
            raise ValueError(
                f"{self.__class__.__name__} can only be used in predict stage"
            )

    def on_predict_start(
        self, trainer: Trainer, pl_module: BiEncoderModule | CrossEncoderModule
    ) -> List[QueryDataset] | List[RunDataset]:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        if self.save_dir is None:
            default_save_dir = Path(pl_module.config.name_or_path)
            if default_save_dir.exists():
                self.save_dir = default_save_dir / "runs"
                print(f"Using default save_dir {self.save_dir}")
            else:
                raise ValueError(
                    "No save_dir provided and model_name_or_path is not a path"
                )
        datasets = [dataloader.dataset for dataloader in dataloaders]
        return datasets

    def get_run_path(self, trainer: Trainer, dataset_idx: int) -> Path:
        dataloaders = trainer.predict_dataloaders
        if self.save_dir is None:
            raise ValueError("No save_dir found; call setup before using this method")
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        dataset = dataloaders[dataset_idx].dataset
        if isinstance(dataset, QueryDataset):
            run_file = dataset.dataset_id.replace("/", "-")
        elif isinstance(dataset, RunDataset):
            if dataset.run_path is None:
                run_file = dataset.dataset_id.replace("/", "-")
            else:
                run_file = dataset.run_path.name.split(".")[0]
        run_file_path = self.save_dir / f"{run_file}.run"
        return run_file_path

    def rank(
        self, batch: SearchBatch | RankBatch, output: LightningIROutput
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        raise NotImplementedError("rank method must be implemented in subclass")

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: BiEncoderModule | CrossEncoderModule,
        prediction: BiEncoderOutput | CrossEncoderOutput,
        batch_indices: Optional[Sequence[int]],
        batch: SearchBatch | RankBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        query_ids = pl_module.all_gather(batch.query_ids)
        batch = self.gather(pl_module, batch)
        prediction = self.gather(pl_module, prediction)
        if not trainer.is_global_zero:
            return

        scores, doc_ids, num_docs = self.rank(batch, prediction)
        scores = scores.float().cpu().numpy()

        query_ids = list(
            itertools.chain.from_iterable(
                itertools.repeat(query_id, num)
                for query_id, num in zip(query_ids, num_docs)
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
        run_df["system"] = pl_module.config.model_type
        run_df = run_df[RUN_HEADER]

        run_file_path = self.get_run_path(trainer, dataloader_idx)
        run_file_path.parent.mkdir(exist_ok=True)
        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"

        run_df.to_csv(run_file_path, header=False, index=False, sep="\t", mode=mode)


class ReRankCallback(RankCallback):
    def rank(
        self, batch: RankBatch, output: LightningIROutput
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        scores = output.scores
        if scores is None:
            raise ValueError("Expected output to have scores")
        doc_ids = batch.doc_ids
        scores = scores.view(-1)
        num_docs = [len(_doc_ids) for _doc_ids in doc_ids]
        doc_ids = list(itertools.chain.from_iterable(doc_ids))
        if scores.shape[0] != len(doc_ids):
            raise ValueError("scores and doc_ids must have the same length")
        return scores.view(-1), doc_ids, num_docs


class SearchCallback(RankCallback):
    def __init__(
        self,
        search_config: SearchConfig,
        save_dir: Path | None = None,
        index_dir: Path | None = None,
    ) -> None:
        super().__init__(save_dir)
        self.index_dir = index_dir
        self.search_config = search_config
        self.index_config: SearchConfig
        self.searcher: Searcher

    def rank(
        self, batch: SearchBatch, output: BiEncoderOutput
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        return self.searcher.search(output)

    def on_predict_start(self, trainer: Trainer, pl_module: BiEncoderModule) -> None:
        datasets = super().on_predict_start(trainer, pl_module)
        if not all(isinstance(dataset, QueryDataset) for dataset in datasets):
            raise ValueError("Expected QueryDatasets for searching")
        docs_dataset_ids = [dataset.docs_dataset_id for dataset in datasets]
        if len(set(docs_dataset_ids)) != 1:
            raise ValueError("All QueryDatasets must have the same docs_dataset_id")
        docs_dataset_id = docs_dataset_ids[0]

        index_dir = self.index_dir
        if index_dir is None:
            if Path(pl_module.config.name_or_path).exists():
                index_dir = Path(pl_module.config.name_or_path) / "indexes"
                index_dir = index_dir / docs_dataset_id
                if not index_dir.exists():
                    raise ValueError(f"No index found at {index_dir}")
            else:
                raise ValueError(
                    "No index_dir provided and model_name_or_path is not a path"
                )
        if isinstance(self.search_config, FaissSearchConfig):
            self.searcher = FaissSearcher(index_dir, self.search_config, pl_module)
        elif isinstance(self.search_config, SparseSearchConfig):
            self.searcher = SparseSearcher(index_dir, self.search_config, pl_module)
        else:
            raise ValueError(
                f"Unknown search config type {self.search_config.__class__.__name__}"
            )
