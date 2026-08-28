"""Module containing callbacks for indexing, searching, ranking, and registering custom datasets."""

from __future__ import annotations

import csv
import gc
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

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

    formatted_number = f"{number:.2f}"

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
            index_dir = index_dir / dataset.dashed_docs_dataset_id
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
                    trainer.print(f"`{save_path}` already exists. set overwrite=True to overwrite")
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

    def _log_to_pg(self, info: dict[str, Any], trainer: Trainer):
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
            dataloader_idx (int | None): Index of the dataloader. Defaults to 0.
        """
        if not trainer.is_global_zero:
            return
        if batch_idx == 0:
            if hasattr(self, "indexer"):
                self.indexer.save()
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

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (BiEncoderModule): LightningIR bi-encoder module.
            outputs (BiEncoderOutput): Encoded documents.
            batch (Any): Batch of input data.
            batch_idx (int): Index of batch in the current dataset.
            dataloader_idx (int | None): Index of the dataloader. Defaults to 0.
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

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        """Hook to save the index after indexing is done.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (LightningIRModule): LightningIR module.
        """
        if not trainer.is_global_zero:
            return
        if hasattr(self, "indexer"):
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
        self.run_dfs: list[pd.DataFrame] = []
        self._current_dataloader_idx: int = 0

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

    def _rank(self, batch: RankBatch, output: LightningIROutput) -> tuple[torch.Tensor, list[str], list[int]]:
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

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningIRModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Hook to write run file.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (LightningIRModule): LightningIR module.
            batch (Any): Batch of input data.
            batch_idx (int): Index of batch in the current dataset.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        if not trainer.is_global_zero:
            return
        if batch_idx == 0:
            if self.run_dfs:
                self._write_run_dfs(trainer, pl_module, self._current_dataloader_idx)
            self.run_dfs = []
            self._current_dataloader_idx = dataloader_idx
        super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

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
            dataloader_idx (int | None): Index of the dataloader. Defaults to 0.
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

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        """Hook to write remaining run files after ranking is done.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer.
            pl_module (LightningIRModule): LightningIR module.
        """
        if not trainer.is_global_zero:
            return
        if self.run_dfs:
            self._write_run_dfs(trainer, pl_module, self._current_dataloader_idx)


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
    ) -> tuple[torch.Tensor, list[str], list[int]]:
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
            dataloader_idx (int | None): Index of the dataloader. Defaults to 0.
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
        qrels_defs: dict[int, str] | None = None,
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
            qrels_defs (dict[int, str] | None): Optional dictionary describing the relevance levels of the qrels.
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


class MvrViewCollapseCallback(Callback):
    """Logs pairwise cosine similarity between MVR VIE token embeddings to diagnose view collapse.

    Logs three metrics, all computed as the mean off-diagonal entry of the pairwise cosine
    similarity matrix over the N=num_viewer_tokens views (1.0 = total collapse, 0.0 = orthogonal):

    - ``train/vie_cos_pre_proj``: backbone hidden states at VIE positions (isolates RoPE / attention).
    - ``train/vie_cos_post_proj``: after the shared linear projection (isolates projection collapse).
    - ``train/vie_cos_post_norm``: after L2 normalization — the vectors actually used in scoring.

    Requires the MVR model's ``encode`` to stash views on ``self._debug_views`` when
    ``self._debug_track_views`` is True (see ``MvrModel.encode``).
    """

    def __init__(self, log_every_n_steps: int = 50, gaussian_perturb_std: float = 0.0):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.gaussian_perturb_std = gaussian_perturb_std

    def on_train_start(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        pl_module.model._debug_track_views = True

        try:
            n = pl_module.config.num_viewer_tokens
            tokenizer = pl_module.tokenizer
            vie_ids = [tokenizer.viewer_token_id(i) for i in range(n)]
            if any(vid is None for vid in vie_ids):
                return
            emb_module = pl_module.model.get_input_embeddings()

            # Optional in-place Gaussian perturbation: break symmetry between the N
            # VIE token rows. mean_resizing in HF initializes all newly-added rows
            # with the same vector, leaving the model with no diversity to bootstrap
            # from. Adding small i.i.d. Gaussian noise per row is the cheapest fix.
            # Noise std is scaled by the row's L2 norm so it's invariant to base
            # embedding magnitude.
            if self.gaussian_perturb_std > 0:
                with torch.no_grad():
                    base = emb_module.weight[vie_ids[0]].clone()
                    base_norm = base.norm().clamp(min=1e-6)
                    noise = torch.randn(
                        len(vie_ids),
                        emb_module.weight.shape[1],
                        device=emb_module.weight.device,
                        dtype=emb_module.weight.dtype,
                    ) * self.gaussian_perturb_std * base_norm
                    for k, vid in enumerate(vie_ids):
                        emb_module.weight[vid] = base + noise[k]
                print(f"[MVR debug] applied Gaussian perturbation std={self.gaussian_perturb_std} to VIE rows")

            vie_rows = emb_module.weight[vie_ids].detach()
            v = torch.nn.functional.normalize(vie_rows.float(), dim=-1)
            sim = v @ v.transpose(-1, -2)
            mask = ~torch.eye(sim.shape[-1], dtype=torch.bool, device=sim.device)
            mean_off = sim[mask].mean()
            print("=" * 80)
            print(f"[MVR debug] VIE input-embedding pairwise cosine (N={sim.shape[-1]}):")
            for row in sim.tolist():
                print("  " + "  ".join(f"{x:+.3f}" for x in row))
            print(f"  mean off-diagonal = {float(mean_off):.4f}")
            print("=" * 80)
        except Exception as e:
            print(f"[MVR debug] could not compute VIE input-embedding cosine: {e}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        pl_module.model._debug_track_views = False

    @staticmethod
    def _mean_off_diag_cos(views: torch.Tensor) -> torch.Tensor:
        v = torch.nn.functional.normalize(views.float(), dim=-1)
        sim = v @ v.transpose(-1, -2)
        n = sim.shape[-1]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        return sim[:, mask].mean()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningIRModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        debug = getattr(pl_module.model, "_debug_views", None)
        if not debug:
            return
        for name, tensor in debug.items():
            pl_module.log(f"train/vie_cos_{name}", self._mean_off_diag_cos(tensor), on_step=True)


class SparsityCollapseCallback(Callback):
    """Stops training when a sparse (SPLADE-style) representation collapses to near-empty vectors.

    SPLADE's ``relu`` sparsification is effectively one-way at the extreme: once a vocabulary
    dimension's logit is negative for *every* input in the batch it receives no gradient, so a
    representation that has decayed to a handful of live terms cannot recover. Run ``v5kd0g11``
    (NeoBERT) reached exactly one nonzero dimension per vector at step 7999 and then burned 13650
    further steps going nowhere, because a degenerate vector trivially minimizes the FLOPS penalty
    while ``MarginMSE`` sits at the score for predicting nothing.

    This callback watches the ``query_num_nonzero`` / ``doc_num_nonzero`` metrics that
    :class:`~lightning_ir.bi_encoder.bi_encoder_module.BiEncoderModule` already logs whenever
    ``sparsification_strategy`` is set, and requests a graceful stop once either stays below its
    floor for ``patience`` consecutive checks. Stopping via ``trainer.should_stop`` (rather than
    raising) lets ``ModelCheckpoint`` write ``last.ckpt``, so the pre-collapse weights survive.

    Read-only with respect to the model: it never modifies parameters or gradients.
    """

    def __init__(
        self,
        min_query_nonzero: float = 5.0,
        min_doc_nonzero: float = 20.0,
        patience: int = 3,
        check_every_n_steps: int = 50,
        warmup_steps: int = 200,
    ) -> None:
        """Initializes the sparsity collapse guard.

        Args:
            min_query_nonzero (float): Floor for the mean number of nonzero query dimensions.
                Defaults to 5.0.
            min_doc_nonzero (float): Floor for the mean number of nonzero document dimensions.
                Defaults to 20.0.
            patience (int): Number of consecutive checks below a floor before stopping. Defaults to 3.
            check_every_n_steps (int): How often to check, in optimizer steps. Defaults to 50.
            warmup_steps (int): Number of initial steps to ignore, so a model that has not yet
                sparsified from its dense initialization is not mistaken for a collapsed one.
                Defaults to 200.
        """
        super().__init__()
        self.min_query_nonzero = min_query_nonzero
        self.min_doc_nonzero = min_doc_nonzero
        self.patience = patience
        self.check_every_n_steps = check_every_n_steps
        self.warmup_steps = warmup_steps
        self._strikes = 0
        self._last_checked = -1

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningIRModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        step = trainer.global_step
        # Elapsed-since-last-check rather than ``step % n``: ``on_train_batch_end`` fires once per
        # microbatch, so with gradient accumulation ``global_step`` repeats and can also skip
        # values -- a modulo test silently never matches for some cadences.
        if step < self.warmup_steps or step - self._last_checked < self.check_every_n_steps:
            return
        self._last_checked = step

        metrics = trainer.callback_metrics
        query_nonzero = metrics.get("query_num_nonzero")
        doc_nonzero = metrics.get("doc_num_nonzero")
        if query_nonzero is None or doc_nonzero is None:
            return  # not a sparse model, or sparsification_strategy is None
        query_nonzero = float(query_nonzero)
        doc_nonzero = float(doc_nonzero)

        if query_nonzero >= self.min_query_nonzero and doc_nonzero >= self.min_doc_nonzero:
            self._strikes = 0
            return

        self._strikes += 1
        print(
            f"[sparsity guard] step {step}: query_num_nonzero={query_nonzero:.1f} "
            f"(floor {self.min_query_nonzero}), doc_num_nonzero={doc_nonzero:.1f} "
            f"(floor {self.min_doc_nonzero}) -- strike {self._strikes}/{self.patience}"
        )
        if self._strikes >= self.patience:
            print(
                f"[sparsity guard] representation collapsed at step {step}; stopping. "
                "The sparsity penalty is too strong for this backbone, or the ranking loss "
                "is not sparsifying it -- see docs on run v5kd0g11."
            )
            trainer.should_stop = True


class MeanValidationMetricCallback(Callback):
    """Logs the mean of one evaluation metric across the validation datasets.

    :meth:`~lightning_ir.base.module.LightningIRModule.validation_step` logs every metric once per
    dataset, e.g. ``msmarco-passage/trec-dl-2019/judged/nDCG@10/dataloader_idx_0``. Lightning's
    :class:`~lightning.pytorch.callbacks.ModelCheckpoint` can only monitor a single key, so
    selecting the checkpoint that is best *on average* over several datasets requires that average
    to exist as a logged scalar in its own right. This callback computes it.

    It logs in ``on_validation_epoch_end``, which Lightning runs after the per-dataset metrics have
    been reduced to epoch values and before ``ModelCheckpoint.on_validation_end``, so the mean is
    always available to the checkpoint monitor within the same validation pass.
    """

    def __init__(
        self,
        metric: str = "nDCG@10",
        name: str | None = None,
        datasets: Sequence[str] | None = None,
    ) -> None:
        """Initializes the callback.

        Args:
            metric (str): Name of the per-dataset metric to average, as it appears after the dataset
                id, e.g. ``nDCG@10``. Defaults to ``"nDCG@10"``.
            name (str | None): Key to log the mean under; this is what ``ModelCheckpoint`` monitors.
                Defaults to ``f"val_mean_{metric}"``.
            datasets (Sequence[str] | None): Dataset ids to average over. If given, every one of them
                must report the metric or a ``ValueError`` is raised — this keeps the monitored
                quantity from silently changing meaning when a dataset drops out. If ``None``, every
                dataset reporting the metric is averaged.
        """
        super().__init__()
        self.metric = metric
        self.name = f"val_mean_{metric}" if name is None else name
        self.datasets = None if datasets is None else list(datasets)

    def _collect(self, callback_metrics: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Picks the per-dataset values of ``self.metric`` out of the logged metrics.

        Args:
            callback_metrics (Mapping[str, Any]): The trainer's callback metrics.
        Returns:
            dict[str, torch.Tensor]: Mapping of dataset id to metric value.
        """
        values: dict[str, torch.Tensor] = {}
        for key, value in callback_metrics.items():
            parts = key.split("/")
            if parts[-1].startswith("dataloader_idx_"):
                parts = parts[:-1]
            if len(parts) < 2 or parts[-1] != self.metric:
                continue
            dataset_id = "/".join(parts[:-1])
            if self.datasets is not None and dataset_id not in self.datasets:
                continue
            values[dataset_id] = torch.as_tensor(value, dtype=torch.float32)
        return values

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        """Logs the mean of ``self.metric`` over the validation datasets.

        Args:
            trainer (Trainer): Lightning trainer.
            pl_module (LightningIRModule): Lightning IR module.
        """
        if trainer.sanity_checking:
            return
        values = self._collect(trainer.callback_metrics)
        if self.datasets is not None:
            missing = [dataset for dataset in self.datasets if dataset not in values]
            if missing:
                raise ValueError(
                    f"No '{self.metric}' was logged for {missing}. Either the dataset ids do not "
                    f"match `inference_datasets`, or `evaluation_metrics` does not include the "
                    f"metric. Logged keys: {sorted(trainer.callback_metrics)}"
                )
        if not values:
            return
        pl_module.log(self.name, torch.stack(list(values.values())).mean())
