"""Module containing callbacks for indexing, searching, ranking, and registering custom datasets."""

from __future__ import annotations

import csv
import gc
import itertools
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, TypeVar

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, TQDMProgressBar

from ..base.validation_utils import evaluate_run
from ..data import LightningIRDataModule, RankBatch, SearchBatch
from ..data.dataset import RUN_HEADER, DocDataset, IRDataset, QueryDataset, RunDataset, _DummyIterableDataset
from ..data.external_datasets.ir_datasets_utils import register_new_dataset
from ..retrieve import IndexConfig, Indexer, SearchConfig, Searcher

if TYPE_CHECKING:
    from ..base import LightningIRModule, LightningIROutput
    from ..bi_encoder import BiEncoderModule, BiEncoderOutput

T = TypeVar("T")


def _format_large_number(number: float) -> str:
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
    """Mixin to gather dataclasses across all processes"""

    def _gather(self, pl_module: LightningIRModule, dataclass: T) -> T:
        if is_dataclass(dataclass):
            return dataclass.__class__(
                **{k: self._gather(pl_module, getattr(dataclass, k)) for k in dataclass.__dataclass_fields__}
            )
        return pl_module.all_gather(dataclass)


class _IndexDirMixin:
    """Mixin to get index_dir"""

    index_dir: Path | str | None
    index_name: str | None

    def _get_index_dir(self, pl_module: BiEncoderModule, dataset: DocDataset) -> Path:
        index_dir = self.index_dir
        if index_dir is None:
            default_index_dir = Path(pl_module.config.name_or_path)
            if default_index_dir.exists():
                index_dir = default_index_dir / "indexes"
            else:
                raise ValueError("No index_dir provided and model_name_or_path is not a path")
        index_dir = Path(index_dir)
        if self.index_name is None:
            index_dir = index_dir / dataset.docs_dataset_id
        else:
            index_dir = index_dir / self.index_name
        return index_dir


class _OverwriteMixin:
    """Mixin to skip datasets (for indexing or searching) if they already exist"""

    _get_save_path: Callable[[LightningIRModule, IRDataset], Path]

    def _remove_overwrite_datasets(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        overwrite = getattr(self, "overwrite", False)
        if not overwrite:
            datamodule: LightningIRDataModule | None = getattr(trainer, "datamodule", None)
            if datamodule is None:
                raise ValueError("No datamodule found")
            if datamodule.inference_datasets is None:
                return
            inference_datasets = list(datamodule.inference_datasets)
            for dataset in inference_datasets:
                save_path = self._get_save_path(pl_module, dataset)
                if save_path.exists():
                    dataset._SKIP = True
                    trainer.print(f"`{save_path}` already exists. Set overwrite=True to overwrite")
                    if (
                        save_path.name.endswith(".run")
                        and dataset.qrels is not None
                        and pl_module.evaluation_metrics is not None
                    ):
                        run = RunDataset._load_csv(save_path)
                        qrels = dataset.qrels.stack(future_stack=True).dropna().astype(int).reset_index()
                        if isinstance(dataset, RunDataset) and dataset.run_path is not None:
                            dataset_id = dataset.run_path.name
                        else:
                            dataset_id = dataset.dataset_id
                        for key, value in evaluate_run(run, qrels, pl_module.evaluation_metrics).items():
                            key = f"{dataset_id}/{key}"
                            pl_module._additional_log_metrics[key] = value

    def _cleanup(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        # reset skip flat and additional log metrics
        datamodule: LightningIRDataModule | None = getattr(trainer, "datamodule", None)
        if datamodule is not None and datamodule.inference_datasets is not None:
            for dataset in datamodule.inference_datasets:
                dataset._SKIP = False
        pl_module._additional_log_metrics = {}


class IndexCallback(Callback, _GatherMixin, _IndexDirMixin, _OverwriteMixin):
    def __init__(
        self,
        index_config: IndexConfig,
        index_dir: Path | str | None = None,
        index_name: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """Callback to index documents using an :py:class:`~lightning_ir.retrieve.base.indexer.Indexer`.

        Args:
            index_config (IndexConfig): Configuration for the indexer.
            index_dir (Path | str | None): Directory to save index(es) to. If None, indexes will be stored in the
                model's directory. Defaults to None.
            index_name (str | None): Name of the index. If None, the dataset's dataset_id or file name will be used.
                Defaults to None.
            overwrite (bool): Whether to skip or overwrite already existing indexes. Defaults to False.
            verbose (bool): Toggle verbose output. Defaults to False.
        """
        super().__init__()
        self.index_config = index_config
        self.index_dir = index_dir
        self.index_name = index_name
        self.overwrite = overwrite
        self.verbose = verbose
        self.indexer: Indexer

    def setup(self, trainer: Trainer, pl_module: BiEncoderModule, stage: str) -> None:
        """Hook to setup the callback.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (BiEncoderModule): LightningIR bi-encoder module used for indexing.
            stage (str): Stage of the trainer, must be "test".
        Raises:
            ValueError: If the stage is not "test".
        """
        if stage != "test":
            raise ValueError(f"{self.__class__.__name__} can only be used in test stage")
        self._remove_overwrite_datasets(trainer, pl_module)

    def teardown(self, trainer: Trainer, pl_module: BiEncoderModule, stage: str) -> None:
        """Hook to cleanup the callback.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (BiEncoderModule): LightningIR bi-encoder module used for indexing.
            stage (str): Stage of the trainer.
        """
        self._cleanup(trainer, pl_module)

    def _get_save_path(self, pl_module: BiEncoderModule, dataset: IRDataset) -> Path:
        if not isinstance(dataset, DocDataset):
            raise ValueError("Expected DocDataset for indexing")
        return self._get_index_dir(pl_module, dataset)

    def _get_indexer(self, pl_module: BiEncoderModule, dataloader_idx: int) -> Indexer:
        dataset = pl_module.get_dataset(dataloader_idx)
        if dataset is None:
            raise ValueError("No dataset found to index")
        if not isinstance(dataset, DocDataset):
            raise ValueError("Expected DocDataset for indexing")
        index_dir = self._get_save_path(pl_module, dataset)

        indexer = self.index_config.indexer_class(index_dir, self.index_config, pl_module, self.verbose)
        return indexer

    def _log_to_pg(self, info: Dict[str, Any], trainer: Trainer):
        pg_callback = trainer.progress_bar_callback
        if pg_callback is None or not isinstance(pg_callback, TQDMProgressBar):
            return
        pg = pg_callback.test_progress_bar
        info = {k: _format_large_number(v) for k, v in info.items()}
        if pg is not None:
            pg.set_postfix(info)

    def on_test_start(self, trainer: Trainer, pl_module: BiEncoderModule) -> None:
        """Hook to test datasets are configured correctly.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (BiEncoderModule): LightningIR bi-encoder module.
        Raises:
            ValueError: If no test_dataloaders are found.
            ValueError: If not all test datasets are :py:class:`~lightning_ir.data.dataset.DocDataset`.
        """
        dataloaders = trainer.test_dataloaders
        if dataloaders is None:
            raise ValueError("No test_dataloaders found")
        datasets = [dataloader.dataset for dataloader in dataloaders]
        if not all(isinstance(dataset, (DocDataset, _DummyIterableDataset)) for dataset in datasets):
            raise ValueError("Expected DocDatasets for indexing")

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: BiEncoderModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Hook to setup the indexer between datasets.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (BiEncoderModule): LightningIR bi-encoder module.
            batch (Any): Batch of input data.
            batch_idx (int): Index of batch in the current dataset.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        if batch_idx == 0:
            self.indexer = self._get_indexer(pl_module, dataloader_idx)
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
        """Hook to pass encoded documents to the indexer

        :param trainer: PyTorch Lightning Trainer
        :type trainer: Trainer
        :param pl_module: LightningIR bi-encoder module
        :type pl_module: BiEncoderModule
        :param outputs: Encoded documents
        :type outputs: BiEncoderOutput
        :param batch: Batch of input data
        :type batch: Any
        :param batch_idx: Index of batch in the current dataset
        :type batch_idx: int
        :param dataloader_idx: Index of the dataloader, defaults to 0
        :type dataloader_idx: int, optional
        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (BiEncoderModule): LightningIR bi-encoder module.
            outputs (BiEncoderOutput): Encoded documents.
            batch (Any): Batch of input data.
            batch_idx (int): Index of batch in the current dataset.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        batch = self._gather(pl_module, batch)
        outputs = self._gather(pl_module, outputs)

        if not trainer.is_global_zero:
            return

        self.indexer.add(batch, outputs)
        self._log_to_pg(
            {
                "num_docs": self.indexer.num_docs,
                "num_embeddings": self.indexer.num_embeddings,
            },
            trainer,
        )
        # TODO if dataset length cannot be inferred, num_test_batches is inf and no index is saved
        if batch_idx == trainer.num_test_batches[dataloader_idx] - 1:
            assert hasattr(self, "indexer")
            self.indexer.save()


class RankCallback(Callback, _GatherMixin, _OverwriteMixin):
    def __init__(
        self, save_dir: Path | str | None = None, run_name: str | None = None, overwrite: bool = False
    ) -> None:
        """Callback to write run file of ranked documents to disk.

        Args:
            save_dir (Path | str | None): Directory to save run files to. If None, run files will be saved in the
                models' directory. Defaults to None.
            run_name (str | None): Name of the run file. If None, the dataset's dataset_id or file name will be used.
                Defaults to None.
            overwrite (bool): Whether to skip or overwrite already existing run files. Defaults to False.
        """
        super().__init__()
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.run_name = run_name
        self.overwrite = overwrite
        self.run_dfs: List[pd.DataFrame] = []

    def setup(self, trainer: Trainer, pl_module: LightningIRModule, stage: str) -> None:
        """Hook to setup the callback.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (LightningIRModule): LightningIR module.
            stage (str): Stage of the trainer, must be "test".
        Raises:
            ValueError: If the stage is not "test".
            ValueError: If no save_dir is provided and model_name_or_path is not a path (the model is not local).
        """
        if stage != "test":
            raise ValueError(f"{self.__class__.__name__} can only be used in test stage")
        if self.save_dir is None:
            default_save_dir = Path(pl_module.config.name_or_path)
            if default_save_dir.exists():
                self.save_dir = default_save_dir / "runs"
                print(f"Using default save_dir `{self.save_dir}` to save runs")
            else:
                raise ValueError("No save_dir provided and model_name_or_path is not a path")
        self._remove_overwrite_datasets(trainer, pl_module)

    def teardown(self, trainer: Trainer, pl_module: BiEncoderModule, stage: str) -> None:
        """Hook to cleanup the callback.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (LightningIRModule): LightningIR bi-encoder module used for indexing.
            stage (str): Stage of the trainer, must be "test".
        """
        self._cleanup(trainer, pl_module)

    def _get_save_path(self, pl_module: LightningIRModule, dataset: IRDataset) -> Path:
        if self.save_dir is None:
            raise ValueError("No save_dir found; call setup before using this method")
        if self.run_name is not None:
            run_file = self.run_name
        elif isinstance(dataset, QueryDataset):
            run_file = f"{dataset.dataset_id.replace('/', '-')}.run"
        elif isinstance(dataset, RunDataset):
            if dataset.run_path is None:
                run_file = f"{dataset.dataset_id.replace('/', '-')}.run"
            else:
                run_file = f"{dataset.run_path.name.split('.')[0]}.run"
        else:
            raise ValueError("Expected QueryDataset or RunDataset for ranking")
        run_file_path = self.save_dir / run_file
        return run_file_path

    def _rank(self, batch: RankBatch, output: LightningIROutput) -> Tuple[torch.Tensor, List[str], List[int]]:
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

    def _write_run_dfs(self, trainer: Trainer, pl_module: LightningIRModule, dataloader_idx: int):
        if not trainer.is_global_zero or not self.run_dfs:
            return
        dataloaders = trainer.test_dataloaders
        if dataloaders is None:
            raise ValueError("No test_dataloaders found")
        dataset = pl_module.get_dataset(dataloader_idx)
        if dataset is None:
            raise ValueError("No dataset found to write run file")
        if not isinstance(dataset, (QueryDataset, RunDataset)):
            raise ValueError("Expected QueryDataset or RunDataset for ranking")
        run_file_path = self._get_save_path(pl_module, dataset)
        run_file_path.parent.mkdir(parents=True, exist_ok=True)
        run_df = pd.concat(self.run_dfs, ignore_index=True)
        run_df.to_csv(run_file_path, header=False, index=False, sep="\t", quoting=csv.QUOTE_NONE)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningIRModule,
        outputs: LightningIROutput,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to aggregate and write ranking to file.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (LightningIRModule): LightningIR module.
            outputs (LightningIROutput): Scored query documents pairs.
            batch (Any): Batch of input data.
            batch_idx (int): Index of batch in the current dataset.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        Raises:
            ValueError: If the batch does not have query_ids.
        """
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        batch = self._gather(pl_module, batch)
        outputs = self._gather(pl_module, outputs)
        if not trainer.is_global_zero:
            return

        query_ids = batch.query_ids
        if query_ids is None:
            raise ValueError("Expected batch to have query_ids")
        scores, doc_ids, num_docs = self._rank(batch, outputs)
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
            self._write_run_dfs(trainer, pl_module, dataloader_idx)
            self.run_dfs = []


class SearchCallback(RankCallback, _IndexDirMixin):
    def __init__(
        self,
        search_config: SearchConfig,
        index_dir: Path | str | None = None,
        index_name: str | None = None,
        save_dir: Path | str | None = None,
        run_name: str | None = None,
        overwrite: bool = False,
        use_gpu: bool = True,
    ) -> None:
        """Callback to which uses index to retrieve documents efficiently.

        Args:
            search_config (SearchConfig): Configuration of the :py:class:`~lightning_ir.retrieve.base.searcher.Searcher`
            index_dir (Path | str | None): Directory where indexes are stored. Defaults to None.
            index_name (str | None): Name of the index. If None, the dataset's dataset_id or file name will be used.
                Defaults to None.
            save_dir (Path | str | None): Directory to save run files to. If None, run files will be saved in the
                model's directory. Defaults to None.
            run_name (str | None): Name of the run file. If None, the dataset's dataset_id or file name will be used.
                Defaults to None.
            overwrite (bool): Whether to skip or overwrite already existing run files. Defaults to False.
            use_gpu (bool): Toggle to use GPU for retrieval. Defaults to True.
        """
        super().__init__(save_dir=save_dir, run_name=run_name, overwrite=overwrite)
        self.search_config = search_config
        self.index_dir = index_dir
        self.index_name = index_name
        self.overwrite = overwrite
        self.use_gpu = use_gpu
        self.searcher: Searcher

    def _get_searcher(self, trainer: Trainer, pl_module: BiEncoderModule, dataset_idx: int) -> Searcher:
        dataloaders = trainer.test_dataloaders
        if dataloaders is None:
            raise ValueError("No test_dataloaders found")
        dataset = dataloaders[dataset_idx].dataset

        index_dir = self._get_index_dir(pl_module, dataset)
        if hasattr(self, "searcher"):
            if self.searcher.index_dir == index_dir:
                return self.searcher
            # free up memory
            del self.searcher
            gc.collect()
            torch.cuda.empty_cache()

        searcher = self.search_config.search_class(index_dir, self.search_config, pl_module, self.use_gpu)
        return searcher

    def _rank(
        self, batch: SearchBatch | RankBatch, output: LightningIROutput
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        if batch.doc_ids is None:
            raise ValueError("BiEncoderModule did not return doc_ids when searching")
        dummy_docs = [[""] * len(ids) for ids in batch.doc_ids]
        batch = RankBatch(batch.queries, dummy_docs, batch.query_ids, batch.doc_ids, batch.qrels)
        return super()._rank(batch, output)

    def on_test_start(self, trainer: Trainer, pl_module: BiEncoderModule) -> None:
        """Hook to validate datasets

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (BiEncoderModule): LightningIR bi-encoder module.
        Raises:
            ValueError: If no test_dataloaders are found.
            ValueError: If not all test datasets are :py:class:`~lightning_ir.data.dataset.QueryDataset`.
        """
        dataloaders = trainer.test_dataloaders
        if dataloaders is None:
            raise ValueError("No test_dataloaders found")
        datasets = [dataloader.dataset for dataloader in dataloaders]
        if not all(isinstance(dataset, (QueryDataset, _DummyIterableDataset)) for dataset in datasets):
            raise ValueError("Expected QueryDatasets for indexing")

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: BiEncoderModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Hook to initialize searcher for new datasets.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (BiEncoderModule): LightningIR bi-encoder module.
            batch (Any): Batch of input data.
            batch_idx (int): Index of the batch in the dataset.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        if batch_idx == 0:
            self.searcher = self._get_searcher(trainer, pl_module, dataloader_idx)
            pl_module.searcher = self.searcher
        super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)


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

        Args:
            dataset_id (str): Dataset id.
            docs (str | None): Path to documents file or valid ir_datasets id from which documents should be taken.
                Defaults to None.
            queries (str | None): Path to queries file or valid ir_datasets id from which queries should be taken.
                Defaults to None.
            qrels (str | None): Path to qrels file or valid ir_datasets id from which qrels will be taken.
                Defaults to None.
            docpairs (str | None): Path to training n-tuple file or valid ir_datasets id from which training tuples
                will be taken. Defaults to None.
            scoreddocs (str | None): Path to run file or valid ir_datasets id from which scored documents will be taken.
                Defaults to None.
            qrels_defs (Dict[int, str] | None): Optional dictionary describing the relevance levels of the qrels.
                Defaults to None.
        """
        super().__init__()
        self.dataset_id = dataset_id
        self.docs = docs
        self.queries = queries
        self.qrels = qrels
        self.docpairs = docpairs
        self.scoreddocs = scoreddocs
        self.qrels_defs = qrels_defs

    def setup(self, trainer: Trainer, pl_module: LightningIRModule, stage: str) -> None:
        """Hook that registers dataset.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (LightningIRModule): Lightning IR module.
            stage (str): Stage of the trainer.
        """
        register_new_dataset(
            self.dataset_id,
            docs=self.docs,
            queries=self.queries,
            qrels=self.qrels,
            docpairs=self.docpairs,
            scoreddocs=self.scoreddocs,
            qrels_defs=self.qrels_defs,
        )
