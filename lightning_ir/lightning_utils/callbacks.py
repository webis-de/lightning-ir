from __future__ import annotations

import itertools
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Tuple, TypeVar

import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback, TQDMProgressBar

from ..data import RankBatch, SearchBatch
from ..data.dataset import RUN_HEADER, DocDataset, QueryDataset, RunDataset
from ..data.ir_datasets_utils import _register_local_dataset
from ..retrieve import IndexConfig, Indexer, SearchConfig, Searcher

if TYPE_CHECKING:
    from ..base import LightningIRModule, LightningIROutput
    from ..bi_encoder import BiEncoderModule, BiEncoderOutput

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


class _GatherMixin:
    def gather(self, pl_module: LightningIRModule, dataclass: T) -> T:
        if is_dataclass(dataclass):
            return dataclass.__class__(
                **{k: self.gather(pl_module, getattr(dataclass, k)) for k in dataclass.__dataclass_fields__}
            )
        return pl_module.all_gather(dataclass)


class _IndexDirMixin:
    index_dir: Path | str | None

    def get_index_dir(self, pl_module: BiEncoderModule, dataset: DocDataset) -> Path:
        index_dir = self.index_dir
        if index_dir is None:
            default_index_dir = Path(pl_module.config.name_or_path)
            if default_index_dir.exists():
                index_dir = default_index_dir / "indexes"
            else:
                raise ValueError("No index_dir provided and model_name_or_path is not a path")
            index_dir = index_dir / dataset.docs_dataset_id
        return Path(index_dir)


class _OverwriteMixin:
    overwrite: bool
    get_save_path: Callable[[Trainer, LightningModule, int], Path]

    def _remove_overwrite_datasets(self, trainer: Trainer, pl_module: LightningIRModule, stage: str) -> None:
        if not self.overwrite:
            datasets = list(trainer.datamodule.inference_datasets)
            remove_datasets = []
            for dataset_idx in range(len(datasets)):
                save_path = self.get_save_path(trainer, pl_module, dataset_idx)
                if save_path.exists():
                    remove_datasets.append(dataset_idx)
                    trainer.print(
                        f"`{save_path}` already exists. Skipping this dataset. Set overwrite=True to overwrite"
                    )
            for dataset_idx in remove_datasets[::-1]:
                del trainer.datamodule.inference_datasets[dataset_idx]


class IndexCallback(Callback, _GatherMixin, _IndexDirMixin, _OverwriteMixin):
    def __init__(
        self,
        index_config: IndexConfig,
        index_dir: Path | str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.index_config = index_config
        self.index_dir = index_dir
        self.overwrite = overwrite
        self.verbose = verbose
        self.indexer: Indexer

    def setup(self, trainer: Trainer, pl_module: BiEncoderModule, stage: str) -> None:
        if stage != "test":
            raise ValueError(f"{self.__class__.__name__} can only be used in test stage")
        self._remove_overwrite_datasets(trainer, pl_module, stage)

    def get_save_path(self, trainer: Trainer, pl_module: BiEncoderModule, dataset_idx: int) -> Path:
        return self.get_index_dir(pl_module, dataset_idx)

    def on_test_start(self, trainer: Trainer, pl_module: BiEncoderModule) -> None:
        dataloaders = trainer.test_dataloaders
        if dataloaders is None:
            raise ValueError("No test_dataloaders found")
        datasets = [dataloader.dataset for dataloader in dataloaders]
        if not all(isinstance(dataset, DocDataset) for dataset in datasets):
            raise ValueError("Expected DocDatasets for indexing")

    def get_indexer(self, trainer: Trainer, pl_module: BiEncoderModule, dataset_idx: int) -> Indexer:
        dataloaders = trainer.test_dataloaders
        if dataloaders is None:
            raise ValueError("No test_dataloaders found")
        dataset = dataloaders[dataset_idx].dataset

        index_dir = self.get_index_dir(pl_module, dataset)

        indexer = self.index_config.indexer_class(index_dir, self.index_config, pl_module.config, self.verbose)
        return indexer

    def log_to_pg(self, info: Dict[str, Any], trainer: Trainer):
        pg_callback = trainer.progress_bar_callback
        if pg_callback is None or not isinstance(pg_callback, TQDMProgressBar):
            return
        pg = pg_callback.test_progress_bar
        info = {k: format_large_number(v) for k, v in info.items()}
        if pg is not None:
            pg.set_postfix(info)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningIRModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if batch_idx == 0:
            self.indexer = self.get_indexer(trainer, pl_module, dataloader_idx)
        super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: BiEncoderModule,
        outputs: BiEncoderOutput,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        batch = self.gather(pl_module, batch)
        outputs = self.gather(pl_module, outputs)

        if not trainer.is_global_zero:
            return

        self.indexer.add(batch, outputs)
        self.log_to_pg(
            {
                "num_docs": self.indexer.num_docs,
                "num_embeddings": self.indexer.num_embeddings,
            },
            trainer,
        )
        if batch_idx == trainer.num_test_batches[dataloader_idx] - 1:
            assert hasattr(self, "indexer")
            self.indexer.save()
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


class RankCallback(BasePredictionWriter, _GatherMixin, _OverwriteMixin):
    def __init__(
        self, save_dir: Path | str | None = None, run_name: str | None = None, overwrite: bool = False
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.run_name = run_name
        self.overwrite = overwrite
        self.run_dfs: List[pd.DataFrame] = []

    def setup(self, trainer: Trainer, pl_module: LightningIRModule, stage: str) -> None:
        if stage != "test":
            raise ValueError(f"{self.__class__.__name__} can only be used in test stage")
        if self.save_dir is None:
            default_save_dir = Path(pl_module.config.name_or_path)
            if default_save_dir.exists():
                self.save_dir = default_save_dir / "runs"
                print(f"Using default save_dir `{self.save_dir}` to save runs")
            else:
                raise ValueError("No save_dir provided and model_name_or_path is not a path")
        self._remove_overwrite_datasets(trainer, pl_module, stage)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.run_dfs = []

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        if trainer.is_global_zero:
            self.write_run_dfs(trainer, pl_module, -1)
            self.run_dfs = []

    def get_save_path(self, trainer: Trainer, pl_module: LightningIRModule, dataset_idx: int) -> Path:
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            raise ValueError("No datamodule found")
        if self.save_dir is None:
            raise ValueError("No save_dir found; call setup before using this method")
        dataset = datamodule.inference_datasets[dataset_idx]
        if self.run_name is not None:
            run_file = self.run_name
        elif isinstance(dataset, QueryDataset):
            run_file = f"{dataset.dataset_id.replace('/', '-')}.run"
        elif isinstance(dataset, RunDataset):
            if dataset.run_path is None:
                run_file = f"{dataset.dataset_id.replace('/', '-')}.run"
            else:
                run_file = f"{dataset.run_path.name.split('.')[0]}.run"
        run_file_path = self.save_dir / run_file
        return run_file_path

    def rank(self, batch: RankBatch, output: LightningIROutput) -> Tuple[torch.Tensor, List[str], List[int]]:
        scores = output.scores
        if scores is None:
            raise ValueError("Expected output to have scores")
        doc_ids = batch.doc_ids
        if doc_ids is None:
            raise ValueError("Expected batch to have doc_ids")
        scores = scores.view(-1)
        num_docs = [len(_doc_ids) for _doc_ids in doc_ids]
        doc_ids = list(itertools.chain.from_iterable(doc_ids))
        if scores.shape[0] != len(doc_ids):
            raise ValueError("scores and doc_ids must have the same length")
        return scores, doc_ids, num_docs

    def write_run_dfs(self, trainer: Trainer, pl_module: LightningIRModule, dataloader_idx: int):
        if not trainer.is_global_zero or not self.run_dfs:
            return
        run_file_path = self.get_save_path(trainer, pl_module, dataloader_idx)
        run_file_path.parent.mkdir(parents=True, exist_ok=True)
        run_df = pd.concat(self.run_dfs, ignore_index=True)
        run_df.to_csv(run_file_path, header=False, index=False, sep="\t")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningIRModule,
        outputs: LightningIROutput,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        batch_indices = trainer.predict_loop.current_batch_indices
        self.write_on_batch_end(trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx)
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningIRModule,
        prediction: LightningIROutput,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = self.gather(pl_module, batch)
        prediction = self.gather(pl_module, prediction)
        if not trainer.is_global_zero:
            return

        query_ids = batch.query_ids
        if query_ids is None:
            raise ValueError("Expected batch to have query_ids")
        scores, doc_ids, num_docs = self.rank(batch, prediction)
        scores = scores.float().cpu().numpy()

        query_ids = list(
            itertools.chain.from_iterable(itertools.repeat(query_id, num) for query_id, num in zip(query_ids, num_docs))
        )
        run_df = pd.DataFrame(zip(query_ids, doc_ids, scores), columns=["query_id", "doc_id", "score"])
        run_df = run_df.sort_values(["query_id", "score"], ascending=[True, False])
        run_df["rank"] = run_df.groupby("query_id")["score"].rank(ascending=False, method="first").astype(int)
        run_df["q0"] = 0
        run_df["system"] = pl_module.model.__class__.__name__
        run_df = run_df[RUN_HEADER]

        self.run_dfs.append(run_df)

        if batch_idx == trainer.num_test_batches[dataloader_idx] - 1:
            self.write_run_dfs(trainer, pl_module, dataloader_idx)
            self.run_dfs = []


class SearchCallback(RankCallback, _IndexDirMixin):
    def __init__(
        self,
        search_config: SearchConfig,
        index_dir: Path | str | None = None,
        save_dir: Path | str | None = None,
        run_name: str | None = None,
        overwrite: bool = False,
        use_gpu: bool = True,
    ) -> None:
        super().__init__(save_dir=save_dir, run_name=run_name, overwrite=overwrite)
        self.search_config = search_config
        self.index_dir = index_dir
        self.overwrite = overwrite
        self.use_gpu = use_gpu
        self.searcher: Searcher

    def on_test_start(self, trainer: Trainer, pl_module: BiEncoderModule) -> None:
        dataloaders = trainer.test_dataloaders
        if dataloaders is None:
            raise ValueError("No test_dataloaders found")
        datasets = [dataloader.dataset for dataloader in dataloaders]
        if not all(isinstance(dataset, QueryDataset) for dataset in datasets):
            raise ValueError("Expected QueryDatasets for indexing")

    def get_searcher(self, trainer: Trainer, pl_module: BiEncoderModule, dataset_idx: int) -> Searcher:
        dataloaders = trainer.test_dataloaders
        if dataloaders is None:
            raise ValueError("No test_dataloaders found")
        dataset = dataloaders[dataset_idx].dataset

        index_dir = self.get_index_dir(pl_module, dataset)

        searcher = self.search_config.search_class(index_dir, self.search_config, pl_module, self.use_gpu)
        return searcher

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningIRModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if batch_idx == 0:
            self.searcher = self.get_searcher(trainer, pl_module, dataloader_idx)
            pl_module.searcher = self.searcher
        return super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def rank(
        self, batch: SearchBatch | RankBatch, output: LightningIROutput
    ) -> Tuple[torch.Tensor | List[str] | List[int]]:
        if batch.doc_ids is None:
            raise ValueError("BiEncoderModule did not return doc_ids when searching")
        dummy_docs = [[""] * len(ids) for ids in batch.doc_ids]
        batch = RankBatch(batch.queries, dummy_docs, batch.query_ids, batch.doc_ids, batch.qrels)
        return super().rank(batch, output)


class ReRankCallback(RankCallback):
    pass


class RegisterLocalDatasetCallback(Callback):

    def __init__(
        self,
        dataset_id: str,
        docs: str | None = None,
        queries: str | None = None,
        qrels: str | None = None,
        docpairs: str | None = None,
        scoreddocs: str | None = None,
        qrels_defs: Dict[int, str] | None = None,
    ):
        """Registers a local dataset with ``ir_datasets``. After registering the dataset, it can be loaded using
        ``ir_datasets.load(dataset_id)``. Currently, the following (optionally gzipped) file types are supported:

        - ``.tsv``, ``.json``, or ``.jsonl`` for documents and queries
        - ``.tsv`` or ``.qrels`` for qrels
        - ``.tsv`` for training n-tuples
        - ``.tsv`` or ``.run`` for scored documents / run files

        :param dataset_id: Dataset id
        :type dataset_id: str
        :param docs: Path to documents file or valid ir_datasets id from which documents should be taken, defaults to None
        :type docs: str | None, optional
        :param queries: Path to queries file or valid ir_datastes id from which queries should be taken, defaults to None
        :type queries: str | None, optional
        :param qrels: Path to qrels file or valid ir_datasets id from which qrels will be taken, defaults to None
        :type qrels: str | None, optional
        :param docpairs: Path to training n-tuple file or valid ir_datasets id from which training tuples will be taken,
            defaults to None
        :type docpairs: str | None, optional
        :param scoreddocs: Path to run file or valid ir_datasets id from which scored documents will be taken,
            defaults to None
        :type scoreddocs: str | None, optional
        :param qrels_defs: Optional dictionary describing the relevance levels of the qrels, defaults to None
        :type qrels_defs: Dict[int, str] | None, optional
        """
        super().__init__()
        self.dataset_id = dataset_id
        self.docs = docs
        self.queries = queries
        self.qrels = qrels
        self.docpairs = docpairs
        self.scoreddocs = scoreddocs
        self.qrels_defs = qrels_defs

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Literal["fit", "validate", "test"]) -> None:
        _register_local_dataset(
            self.dataset_id, self.docs, self.queries, self.qrels, self.docpairs, self.scoreddocs, self.qrels_defs
        )
