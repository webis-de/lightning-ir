"""
Datasets for Lightning IR that data loading and sampling.

This module defines several datasets that handle loading and sampling data for training and inference.
"""

import csv
import warnings
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Any, Literal, Self

import ir_datasets
import numpy as np
import pandas as pd
import torch
from ir_datasets.formats import GenericDoc, GenericDocPair
from torch.distributed import get_rank, get_world_size
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .data import DocSample, QuerySample, RankSample
from .external_datasets.ir_datasets_utils import ScoredDocTuple
from .external_datasets.lsr_benchmark import register_lsr_dataset_to_ir_datasets

RUN_HEADER = ["query_id", "q0", "doc_id", "rank", "score", "system"]


class _DummyIterableDataset(IterableDataset):
    """Dummy iterable dataset to use when all inference datasets are skipped."""

    def __iter__(self) -> Iterator:
        yield from iter([])


class IRDataset:
    _SKIP: bool = False
    """set to True to skip the dataset during inference."""

    def __init__(self, dataset: str) -> None:
        """Initializes a new IRDataset.

        Args:
            dataset (str): Dataset name.
        """
        super().__init__()
        self._dataset = dataset
        self._queries = None
        self._docs = None
        self._qrels = None

    @property
    def dataset(self) -> str:
        """Dataset name.

        Returns:
            str: Dataset name.
        """
        return self.DASHED_DATASET_MAP.get(self._dataset, self._dataset)

    @property
    def dataset_id(self) -> str:
        """Dataset id.

        Returns:
            str: Dataset id.
        """
        if self.ir_dataset is None:
            return self.dataset
        return self.ir_dataset.dataset_id()

    @property
    def docs_dataset_id(self) -> str:
        """ID of the dataset containing the documents.

        Returns:
            str: Document dataset id.
        """
        return ir_datasets.docs_parent_id(self.dataset_id)

    @property
    def dashed_docs_dataset_id(self) -> str:
        """Dataset id with dashes instead of slashes for the documents dataset.

        Returns:
            str: Document dataset id with dashes.
        """
        return self.docs_dataset_id.replace("/", "-")

    @property
    def ir_dataset(self) -> ir_datasets.Dataset | None:
        """Instance of ir_datasets.Dataset.

        Returns:
            ir_datasets.Dataset | None: Instance of ir_datasets.Dataset or None if the dataset is not found.
        """
        if self.dataset and self.dataset.startswith("lsr-benchmark/"):
            register_lsr_dataset_to_ir_datasets(self.dataset)

        try:
            return ir_datasets.load(self.dataset)
        except KeyError:
            return None

    @property
    def DASHED_DATASET_MAP(self) -> dict[str, str]:
        """Map of dataset names with dashes to dataset names with slashes.

        Returns:
            dict[str, str]: Dataset map.
        """
        return {dataset.replace("/", "-"): dataset for dataset in ir_datasets.registry._registered}

    @property
    def queries(self) -> pd.Series:
        """Queries in the dataset.

        Returns:
            pd.Series: Queries.
        Raises:
            ValueError: If no queries are found in the dataset.
        """
        if self._queries is None:
            if self.ir_dataset is None:
                raise ValueError(f"Unable to find dataset {self.dataset} in ir-datasets")
            queries_iter = self.ir_dataset.queries_iter()
            self._queries = pd.Series(
                {query.query_id: query.default_text() for query in queries_iter},
                name="text",
            )
            self._queries.index.name = "query_id"
        return self._queries

    @property
    def docs(self) -> ir_datasets.indices.Docstore | dict[str, GenericDoc]:
        """Documents in the dataset.

        Returns:
            ir_datasets.indices.Docstore | dict[str, GenericDoc]: Documents.
        Raises:
            ValueError: If no documents are found in the dataset.
        """
        if self._docs is None:
            if self.ir_dataset is None:
                raise ValueError(f"Unable to find dataset {self.dataset} in ir-datasets")
            self._docs = self.ir_dataset.docs_store()
        return self._docs

    @property
    def qrels(self) -> pd.DataFrame | None:
        """Qrels in the dataset.

        Returns:
            pd.DataFrame | None: Qrels.
        """
        if self._qrels is not None:
            return self._qrels
        if self.ir_dataset is None or not self.ir_dataset.has_qrels():
            return None
        qrels = pd.DataFrame(self.ir_dataset.qrels_iter()).rename({"subtopic_id": "iteration"}, axis=1)
        if "iteration" not in qrels.columns:
            qrels["iteration"] = 0
        qrels = qrels.drop_duplicates(["query_id", "doc_id", "iteration"])
        qrels = qrels.set_index(["query_id", "doc_id", "iteration"]).unstack(level=-1)
        self._qrels = qrels
        return self._qrels

    def prepare_constituent(self, constituent: Literal["qrels", "queries", "docs", "scoreddocs", "docpairs"]) -> None:
        """Downloads the constituent of the dataset using ir_datasets if needed.

        Args:
            constituent (Literal["qrels", "queries", "docs", "scoreddocs", "docpairs"]): Constituent to download.
        """
        if self.ir_dataset is None:
            return
        if self.ir_dataset.has(constituent):
            if constituent == "docs" and hasattr(self.ir_dataset, "docs_store"):
                # build docs store if not already built
                docs_store = self.ir_dataset.docs_store()
                if not docs_store.built():
                    docs_store.build()
            else:
                # get first item to trigger download
                next(getattr(self.ir_dataset, f"{constituent}_iter")())


class _DataParallelIterableDataset(IterableDataset):
    # https://github.com/Lightning-AI/pytorch-lightning/issues/15734
    def __init__(self) -> None:
        super().__init__()
        # TODO add support for multi-gpu and multi-worker inference; currently
        # doesn't work
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        try:
            world_size = get_world_size()
            process_rank = get_rank()
        except (RuntimeError, ValueError):
            world_size = 1
            process_rank = 0

        self.num_replicas = num_workers * world_size
        self.rank = process_rank * num_workers + worker_id


class QueryDataset(IRDataset, _DataParallelIterableDataset):
    def __init__(self, query_dataset: str, num_queries: int | None = None) -> None:
        """Dataset containing queries.

        Args:
            query_dataset (str): Path to file containing queries or valid ir_datasets id.
            num_queries (int | None, optional): Number of queries in dataset. If None, the number of queries will
                attempted to be inferred. Defaults to None.
        """
        super().__init__(query_dataset)
        super(IRDataset, self).__init__()
        self.num_queries = num_queries

    def __len__(self) -> int | None:
        """Number of queries in the dataset. Returns None if the number of queries cannot be inferred.

        Returns:
            int | None: Number of queries.
        """
        # TODO fix len for multi-gpu and multi-worker inference
        return self.num_queries or getattr(self.ir_dataset, "queries_count", lambda: None)() or None

    def __iter__(self) -> Iterator[QuerySample]:
        """Iterate over queries in the dataset.

        Yields:
            QuerySample: Query sample.
        """
        start = self.rank
        stop = self.num_queries
        step = self.num_replicas
        for sample in islice(self.ir_dataset.queries_iter(), start, stop, step):
            query_sample = QuerySample.from_ir_dataset_sample(sample)
            if self.qrels is not None:
                qrels = (
                    self.qrels.loc[[query_sample.query_id]]
                    .stack(future_stack=True)
                    .dropna()
                    .astype(int)
                    .reset_index()
                    .to_dict(orient="records")
                )
                query_sample.qrels = qrels
            yield query_sample

    def prepare_data(self) -> None:
        """Downloads queries using ir_datasets if needed."""
        self.prepare_constituent("queries")


class DocDataset(IRDataset, _DataParallelIterableDataset):
    def __init__(self, doc_dataset: str, num_docs: int | None = None, text_fields: Sequence[str] | None = None) -> None:
        """Dataset containing documents.

        Args:
            doc_dataset (str): Path to file containing documents or valid ir_datasets id.
            num_docs (int | None, optional): Number of documents in dataset. If None, the number of documents will
                attempted to be inferred. Defaults to None.
            text_fields (Sequence[str] | None, optional): Fields to parse the document text from. Defaults to None.
        """
        super().__init__(doc_dataset)
        super(IRDataset, self).__init__()
        self.num_docs = num_docs
        self.text_fields = text_fields

    def __len__(self) -> int | None:
        """Number of documents in the dataset. Returns None if the number of documents cannot be inferred.

        Returns:
            int | None: Number of documents.
        """
        # TODO fix len for multi-gpu and multi-worker inference
        return self.num_docs or getattr(self.ir_dataset, "docs_count", lambda: None)() or None

    def __iter__(self) -> Iterator[DocSample]:
        """Iterate over documents in the dataset.

        Yields:
            DocSample: Document sample.
        """
        start = self.rank
        stop = self.num_docs
        step = self.num_replicas
        for sample in islice(self.ir_dataset.docs_iter(), start, stop, step):
            yield DocSample.from_ir_dataset_sample(sample, self.text_fields)

    def prepare_data(self) -> None:
        """Downloads documents using ir_datasets if needed."""
        self.prepare_constituent("docs")


class Sampler:
    """Helper class for sampling subsets of documents from a ranked list."""

    @staticmethod
    def single_relevant(documents: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Sampling strategy to randomly sample a single relevant document. The remaining ``sample_size - 1``
        are non-relevant.

        Args:
            documents (pd.DataFrame): Ranked list of documents.
            sample_size (int): Number of documents to sample.
        Returns:
            pd.DataFrame: Sampled documents.
        """
        relevance = documents.filter(like="relevance").max(axis=1).fillna(0)
        relevant = documents.loc[relevance.gt(0)].sample(1)
        non_relevant_bool = relevance.eq(0) & ~documents["rank"].isna()
        num_non_relevant = non_relevant_bool.sum()
        sample_non_relevant = min(sample_size - 1, num_non_relevant)
        non_relevant = documents.loc[non_relevant_bool].sample(sample_non_relevant)
        return pd.concat([relevant, non_relevant])

    @staticmethod
    def top(documents: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Sampling strategy to randomly sample a single relevant document. The remaining ``sample_size - 1``
        are non-relevant.

        Args:
            documents (pd.DataFrame): Ranked list of documents.
            sample_size (int): Number of documents to sample.
        Returns:
            pd.DataFrame: Sampled documents.
        """
        return documents.head(sample_size)

    @staticmethod
    def top_and_random(documents: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Sampling strategy to randomly sample half the ``sample_size`` documents from the top of the ranking and the
        other half randomly.

        Args:
            documents (pd.DataFrame): Ranked list of documents.
            sample_size (int): Number of documents to sample.
        Returns:
            pd.DataFrame: Sampled documents.
        """
        top_size = sample_size // 2
        random_size = sample_size - top_size
        top = documents.head(top_size)
        random = documents.iloc[top_size:].sample(random_size)
        return pd.concat([top, random])

    @staticmethod
    def random(documents: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Sampling strategy to randomly sample ``sample_size`` documents.

        Args:
            documents (pd.DataFrame): Ranked list of documents.
            sample_size (int): Number of documents to sample.
        Returns:
            pd.DataFrame: Sampled documents.
        """
        return documents.sample(sample_size)

    @staticmethod
    def log_random(documents: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Sampling strategy to randomly sample documents with a higher probability to sample documents from the top of
        the ranking.

        Args:
            documents (pd.DataFrame): Ranked list of documents.
            sample_size (int): Number of documents to sample.
        Returns:
            pd.DataFrame: Sampled documents.
        """
        weights = 1 / np.log1p(documents["rank"])
        weights[weights.isna()] = weights.min()
        return documents.sample(sample_size, weights=weights)

    @staticmethod
    def sample(
        df: pd.DataFrame,
        sample_size: int,
        sampling_strategy: Literal["single_relevant", "top", "random", "log_random", "top_and_random"],
    ) -> pd.DataFrame:
        """
        Samples a subset of documents from a ranked list given a sampling_strategy.

        Args:
            documents (pd.DataFrame): Ranked list of documents.
            sample_size (int): Number of documents to sample.
        Returns:
            pd.DataFrame: Sampled documents.
        """
        if sample_size == -1:
            return df
        if hasattr(Sampler, sampling_strategy):
            return getattr(Sampler, sampling_strategy)(df, sample_size)
        raise ValueError("Invalid sampling strategy.")


class RunDataset(IRDataset, Dataset):
    def __init__(
        self,
        run_path_or_id: Path | str,
        depth: int = -1,
        sample_size: int = -1,
        sampling_strategy: Literal["single_relevant", "top", "random", "log_random", "top_and_random"] = "top",
        targets: Literal["relevance", "subtopic_relevance", "rank", "score"] | None = None,
        normalize_targets: bool = False,
        add_docs_not_in_ranking: bool = False,
    ) -> None:
        """Dataset containing a list of queries with a ranked list of documents per query. Subsets of the ranked list
        can be sampled using different sampling strategies.

        Args:
            run_path_or_id (Path | str): Path to a run file or valid ir_datasets id.
            depth (int): Depth at which to cut off the ranking. If -1 the full ranking is kept.
                Defaults to -1.
            sample_size (int): The number of documents to sample per query. Defaults to -1.
            sampling_strategy (Literal["single_relevant", "top", "random", "log_random", "top_and_random"]):
                The sample strategy to use to sample documents. Defaults to "top".
            targets (Literal["relevance", "subtopic_relevance", "rank", "score"] | None):
                The data type to use as targets for a model during fine-tuning. If "relevance" the relevance
                judgements are parsed from qrels. Defaults to None.
            normalize_targets (bool): Whether to normalize the targets between 0 and 1. Defaults to False.
            add_docs_not_in_ranking (bool): Whether to add relevant documents to a sample that are in the qrels but not
                in the ranking. Defaults to False.
        """
        self.run_path = None
        if Path(run_path_or_id).is_file():
            self.run_path = Path(run_path_or_id)
            dataset = self.run_path.name.split(".")[0].split("__")[-1]
        else:
            dataset = str(run_path_or_id)
        super().__init__(dataset)
        self.depth = depth
        self.sample_size = sample_size
        self.sampling_strategy = sampling_strategy
        self.targets = targets
        self.normalize_targets = normalize_targets
        self.add_docs_not_in_ranking = add_docs_not_in_ranking

        if self.sampling_strategy == "top" and self.sample_size > self.depth:
            warnings.warn(
                "Sample size is greater than depth and top sampling strategy is used. "
                "This can cause documents to be sampled that are not contained "
                "in the run file, but that are present in the qrels.",
                stacklevel=2,
            )

        self.run: pd.DataFrame | None = None

    def prepare_data(self) -> Self:
        """Downloads docs, queries, scoreddocs, and qrels using ir_datasets if needed and available."""
        self.prepare_constituent("docs")
        self.prepare_constituent("queries")
        if self.run_path is None:
            self.prepare_constituent("scoreddocs")
            self.prepare_constituent("docpairs")
        self.prepare_constituent("qrels")
        return self

    def _setup(self):
        if self.run is not None:
            return
        self.run = self._load_run()
        self.run = self.run.drop_duplicates(["query_id", "doc_id"])

        if self.qrels is not None:
            run_query_ids = pd.Index(self.run["query_id"].drop_duplicates())
            qrels_query_ids = self.qrels.index.get_level_values("query_id").unique()
            query_ids = run_query_ids.intersection(qrels_query_ids)
            if len(run_query_ids.difference(qrels_query_ids)):
                self.run = self.run[self.run["query_id"].isin(query_ids)]
            # outer join if docs are from ir_datasets else only keep docs in run
            how = "left"
            if self._docs is None and self.add_docs_not_in_ranking:
                how = "outer"
            self.run = self.run.merge(
                self.qrels.loc[pd.IndexSlice[query_ids, :]].droplevel(0, axis=1).add_prefix("relevance_", axis=1),
                on=["query_id", "doc_id"],
                how=how,
            )

        if self.sample_size != -1:
            num_docs_per_query = self.run.groupby("query_id").transform("size")
            self.run = self.run[num_docs_per_query >= self.sample_size]

        self.run = self.run.sort_values(["query_id", "rank"])
        self.run_groups = self.run.groupby("query_id")
        self.query_ids = list(self.run_groups.groups.keys())

        if self.depth != -1 and self.run["rank"].max() < self.depth:
            warnings.warn("Depth is greater than the maximum rank in the run file.", stacklevel=2)

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=RUN_HEADER,
            usecols=[0, 1, 2, 3, 4],
            dtype={"query_id": str, "doc_id": str},
            quoting=csv.QUOTE_NONE,
            na_filter=False,
        )

    @staticmethod
    def _load_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def _load_json(path: Path) -> pd.DataFrame:
        kwargs: dict[str, Any] = {}
        if ".jsonl" in path.suffixes:
            kwargs["lines"] = True
            kwargs["orient"] = "records"
        run = pd.read_json(path, **kwargs)
        return run

    def _get_run_path(self) -> Path | None:
        run_path = self.run_path
        if run_path is None:
            if self.ir_dataset is None or not self.ir_dataset.has_scoreddocs():
                raise ValueError(f"Run file or dataset with scoreddocs required. Got {self._dataset}")
            try:
                run_path = self.ir_dataset.scoreddocs_handler().scoreddocs_path()
            except NotImplementedError:
                pass
        return run_path

    def _clean_run(self, run: pd.DataFrame) -> pd.DataFrame:
        run = run.rename(
            {"qid": "query_id", "docid": "doc_id", "docno": "doc_id", "Q0": "iteration", "q0": "iteration"},
            axis=1,
        )
        dtypes = {"rank": np.int32, "query_id": str, "doc_id": str}
        if "score" in run.columns:
            dtypes["score"] = np.float32
        run = run.astype(dtypes)
        if "query" in run.columns:
            self._queries = run.drop_duplicates("query_id").set_index("query_id")["query"].rename("text")
            run = run.drop("query", axis=1)
        if "text" in run.columns:
            self._docs = (
                run.drop_duplicates("doc_id").set_index("doc_id")["text"].map(lambda x: GenericDoc("", x)).to_dict()
            )
            run = run.drop("text", axis=1)
        if self.depth != -1:
            run = run[run["rank"] <= self.depth]
        return run

    def _load_run(self) -> pd.DataFrame:
        suffix_load_map = {
            ".tsv": self._load_csv,
            ".run": self._load_csv,
            ".csv": self._load_csv,
            ".parquet": self._load_parquet,
            ".json": self._load_json,
            ".jsonl": self._load_json,
        }
        run = None

        # try loading run from file
        run_path = self._get_run_path()
        if run_path is not None:
            load_func = suffix_load_map.get(run_path.suffixes[0], None)
            if load_func is not None:
                try:
                    run = load_func(run_path)
                except Exception:
                    pass

        # try loading run from ir_datasets
        if run is None and self.ir_dataset is not None and self.ir_dataset.has_scoreddocs():
            run = pd.DataFrame(self.ir_dataset.scoreddocs_iter())
            run["rank"] = run.groupby("query_id")["score"].rank("first", ascending=False)
            run = run.sort_values(["query_id", "rank"])

        if run is None:
            raise ValueError("Invalid run file format.")

        run = self._clean_run(run)
        return run

    @property
    def qrels(self) -> pd.DataFrame | None:
        """The qrels in the dataset. If the dataset does not contain qrels, the qrels are None.

        Returns:
            pd.DataFrame | None: Qrels.
        """
        if self._qrels is not None:
            return self._qrels
        if self.run is not None and "relevance" in self.run:
            qrels = self.run[["query_id", "doc_id", "relevance"]].copy()
            if "iteration" in self.run:
                qrels["iteration"] = self.run["iteration"]
            else:
                qrels["iteration"] = "0"
            self.run = self.run.drop(["relevance", "iteration"], axis=1, errors="ignore")
            qrels = qrels.drop_duplicates(["query_id", "doc_id", "iteration"])
            qrels = qrels.set_index(["query_id", "doc_id", "iteration"]).unstack(level=-1)
            self._qrels = qrels
            return self._qrels
        return super().qrels

    def __len__(self) -> int:
        """Number of queries in the dataset.

        Returns:
            int: Number of queries.
        """
        self._setup()
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> RankSample:
        """Samples a single query and corresponding ranked documents from the run. The documents are sampled according
        to the sampling strategy and sample size.

        Args:
            idx (int): Index of the query.
        Returns:
            RankSample: Sampled query and documents.
        Raises:
            ValueError: If the targets are not found in the run file.
        """
        self._setup()
        query_id = str(self.query_ids[idx])
        group = self.run_groups.get_group(query_id).copy()
        query = self.queries[query_id]
        group = Sampler.sample(group, self.sample_size, self.sampling_strategy)

        doc_ids = tuple(group["doc_id"])
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)

        targets = None
        if self.targets is not None:
            filtered = group.set_index("doc_id").loc[list(doc_ids)].filter(like=self.targets).fillna(0)
            if filtered.empty:
                raise ValueError(f"targets `{self.targets}` not found in run file")
            targets = torch.from_numpy(filtered.values)
            if self.targets == "rank":
                # invert ranks to be higher is better (necessary for loss functions)
                targets = self.depth - targets + 1
            if self.normalize_targets:
                targets_min = targets.min()
                targets_max = targets.max()
                targets = (targets - targets_min) / (targets_max - targets_min)
        qrels = None
        if self.qrels is not None:
            qrels = (
                self.qrels.loc[[query_id]]
                .stack(future_stack=True)
                .dropna()
                .astype(int)
                .reset_index()
                .to_dict(orient="records")
            )
        return RankSample(query_id, query, doc_ids, docs, targets, qrels)


class TupleDataset(IRDataset, IterableDataset):
    def __init__(
        self,
        tuples_dataset: str,
        targets: Literal["order", "score"] = "order",
        num_docs: int | None = None,
    ) -> None:
        """Dataset containing tuples of a query and n-documents. Used for fine-tuning models on ranking tasks.

        Args:
            tuples_dataset (str): Path to file containing tuples or valid ir_datasets id.
            targets (Literal["order", "score"], optional): Data type to use as targets for a model during fine-tuning.
                Defaults to "order".
            num_docs (int | None, optional): Maximum number of documents per query. Defaults to None.
        """
        super().__init__(tuples_dataset)
        super(IRDataset, self).__init__()
        self.targets = targets
        self.num_docs = num_docs

    def _parse_sample(
        self, sample: ScoredDocTuple | GenericDocPair
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...] | None]:
        if isinstance(sample, GenericDocPair):
            if self.targets == "score":
                raise ValueError("ScoredDocTuple required for score targets.")
            targets = (1.0, 0.0)
            doc_ids = (sample.doc_id_a, sample.doc_id_b)
        elif isinstance(sample, ScoredDocTuple):
            doc_ids = sample.doc_ids[: self.num_docs]
            if self.targets == "score":
                if sample.scores is None:
                    raise ValueError("tuples dataset does not contain scores")
                targets = sample.scores
            elif self.targets == "order":
                targets = tuple([1.0] + [0.0] * (sample.num_docs - 1))
            else:
                raise ValueError(f"invalid value for targets, got {self.targets}, expected one of (order, score)")
            targets = targets[: self.num_docs]
        else:
            raise ValueError("Invalid sample type.")
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)
        return doc_ids, docs, targets

    def __iter__(self) -> Iterator[RankSample]:
        """Iterates over tuples in the dataset.

        Yields:
            RankSample: Sampled query and documents with targets.
        """
        for sample in self.ir_dataset.docpairs_iter():
            query_id = sample.query_id
            query = self.queries.loc[query_id]
            doc_ids, docs, targets = self._parse_sample(sample)
            if targets is not None:
                targets = torch.tensor(targets)
            yield RankSample(query_id, query, doc_ids, docs, targets)

    def prepare_data(self) -> None:
        """Downloads tuples using ir_datasets if needed."""
        self.prepare_constituent("docs")
        self.prepare_constituent("queries")
        self.prepare_constituent("docpairs")
