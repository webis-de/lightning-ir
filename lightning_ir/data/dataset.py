import warnings
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Tuple

import ir_datasets
import numpy as np
import pandas as pd
import torch
from ir_datasets.formats import GenericDoc, GenericDocPair
from torch.distributed import get_rank, get_world_size
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .data import DocSample, QuerySample, RankSample
from .ir_datasets_utils import ScoredDocTuple

RUN_HEADER = ["query_id", "q0", "doc_id", "rank", "score", "system"]


class IRDataset:
    def __init__(self, dataset: str) -> None:
        super().__init__()
        if dataset in self.DASHED_DATASET_MAP:
            dataset = self.DASHED_DATASET_MAP[dataset]
        self.dataset = dataset
        try:
            self.ir_dataset = ir_datasets.load(dataset)
        except KeyError:
            self.ir_dataset = None
        self._queries = None
        self._docs = None
        self._qrels = None

    @property
    def DASHED_DATASET_MAP(self) -> Dict[str, str]:
        return {dataset.replace("/", "-"): dataset for dataset in ir_datasets.registry._registered}

    @property
    def queries(self) -> pd.Series:
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
    def docs(self) -> ir_datasets.indices.Docstore | Dict[str, GenericDoc]:
        if self._docs is None:
            if self.ir_dataset is None:
                raise ValueError(f"Unable to find dataset {self.dataset} in ir-datasets")
            self._docs = self.ir_dataset.docs_store()
        return self._docs

    @property
    def qrels(self) -> pd.DataFrame | None:
        if self._qrels is not None:
            return self._qrels
        if self.ir_dataset is None:
            return None
        qrels = pd.DataFrame(self.ir_dataset.qrels_iter()).rename({"subtopic_id": "iteration"}, axis=1)
        if "iteration" not in qrels.columns:
            qrels["iteration"] = 0
        qrels = qrels.drop_duplicates(["query_id", "doc_id", "iteration"])
        qrels = qrels.set_index(["query_id", "doc_id", "iteration"]).unstack(level=-1)
        qrels = qrels.droplevel(0, axis=1)
        self._qrels = qrels
        return self._qrels

    @property
    def dataset_id(self) -> str:
        if self.ir_dataset is None:
            return self.dataset
        return self.ir_dataset.dataset_id()

    @property
    def docs_dataset_id(self) -> str:
        return ir_datasets.docs_parent_id(self.dataset_id)


class DataParallelIterableDataset(IterableDataset):
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


class QueryDataset(IRDataset, DataParallelIterableDataset):
    def __init__(self, query_dataset: str, num_queries: int | None = None) -> None:
        super().__init__(query_dataset)
        super(IRDataset, self).__init__()
        self.num_queries = num_queries

    def __len__(self) -> int:
        # TODO fix len for multi-gpu and multi-worker inference
        return self.num_queries or self.ir_dataset.queries_count()

    def __iter__(self) -> Iterator[QuerySample]:
        start = self.rank
        stop = self.num_queries
        step = self.num_replicas
        for sample in islice(self.ir_dataset.queries_iter(), start, stop, step):
            query_sample = QuerySample.from_ir_dataset_sample(sample)
            if self.qrels is not None:
                qrels = (
                    self.qrels.loc[[query_sample.query_id]]
                    .stack()
                    .rename("relevance")
                    .astype(int)
                    .reset_index()
                    .to_dict(orient="records")
                )
                query_sample.qrels = qrels
            yield query_sample


class DocDataset(IRDataset, DataParallelIterableDataset):
    def __init__(self, doc_dataset: str, num_docs: int | None = None) -> None:
        super().__init__(doc_dataset)
        super(IRDataset, self).__init__()
        self.num_docs = num_docs

    def __len__(self) -> int:
        # TODO fix len for multi-gpu and multi-worker inference
        return self.num_docs or self.ir_dataset.docs_count()

    def __iter__(self) -> Iterator[DocSample]:
        start = self.rank
        stop = self.num_docs
        step = self.num_replicas
        for sample in islice(self.ir_dataset.docs_iter(), start, stop, step):
            yield DocSample.from_ir_dataset_sample(sample)


class Sampler:

    @staticmethod
    def single_relevant(group: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        relevance = group.filter(like="relevance").max(axis=1).fillna(0)
        relevant = group.loc[relevance.gt(0)].sample(1)
        non_relevant_bool = relevance.eq(0) & ~group["rank"].isna()
        num_non_relevant = non_relevant_bool.sum()
        sample_non_relevant = min(sample_size - 1, num_non_relevant)
        non_relevant = group.loc[non_relevant_bool].sample(sample_non_relevant)
        return pd.concat([relevant, non_relevant])

    @staticmethod
    def top(group: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        return group.head(sample_size)

    @staticmethod
    def top_and_random(group: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        top_size = sample_size // 2
        random_size = sample_size - top_size
        top = group.head(top_size)
        random = group.iloc[top_size:].sample(random_size)
        return pd.concat([top, random])

    @staticmethod
    def random(group: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        return group.sample(sample_size)

    @staticmethod
    def log_random(group: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        weights = 1 / np.log1p(group["rank"])
        weights[weights.isna()] = weights.min()
        return group.sample(sample_size, weights=weights)

    @staticmethod
    def sample(
        df: pd.DataFrame,
        sample_size: int,
        sampling_strategy: Literal["single_relevant", "top", "random", "log_random", "top_and_random"],
    ) -> pd.DataFrame:
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
        add_non_retrieved_docs: bool = False,
    ) -> None:
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

        if self.sampling_strategy == "top" and self.sample_size > self.depth:
            warnings.warn(
                "Sample size is greater than depth and top sampling strategy is used. "
                "This can cause documents to be sampled that are not contained "
                "in the run file, but that are present in the qrels."
            )

        self.run = self.load_run()
        self.run = self.run.drop_duplicates(["query_id", "doc_id"])

        if self.qrels is not None:
            run_query_ids = pd.Index(self.run["query_id"].drop_duplicates())
            qrels_query_ids = self.qrels.index.get_level_values("query_id").unique()
            query_ids = run_query_ids.intersection(qrels_query_ids)
            if len(run_query_ids.difference(qrels_query_ids)):
                self.run = self.run[self.run["query_id"].isin(query_ids)]
            # outer join if docs are from ir_datasets else only keep docs in run
            how = "left"
            if self._docs is None and add_non_retrieved_docs:
                how = "outer"
            self.run = self.run.merge(
                self.qrels.loc[pd.IndexSlice[query_ids, :]].add_prefix("relevance_", axis=1),
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
            warnings.warn("Depth is greater than the maximum rank in the run file.")

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=RUN_HEADER,
            usecols=[0, 2, 3, 4],
            dtype={"query_id": str, "doc_id": str},
        )

    @staticmethod
    def load_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path).rename(
            {
                "qid": "query_id",
                "docid": "doc_id",
                "docno": "doc_id",
            },
            axis=1,
        )

    @staticmethod
    def load_json(path: Path) -> pd.DataFrame:
        kwargs: Dict[str, Any] = {}
        if ".jsonl" in path.suffixes:
            kwargs["lines"] = True
            kwargs["orient"] = "records"
        run = pd.read_json(
            path,
            **kwargs,
            dtype={
                "query_id": str,
                "qid": str,
                "doc_id": str,
                "docid": str,
                "docno": str,
            },
        )
        return run

    def _get_run_path(self) -> Path | None:
        run_path = self.run_path
        if run_path is None:
            if self.ir_dataset is None or not self.ir_dataset.has_scoreddocs():
                raise ValueError("Run file or dataset with scoreddocs required.")
            try:
                run_path = self.ir_dataset.scoreddocs_handler().scoreddocs_path()
            except NotImplementedError:
                pass
        return run_path

    def _clean_run(self, run: pd.DataFrame) -> pd.DataFrame:
        run = run.rename(
            {"qid": "query_id", "docid": "doc_id", "docno": "doc_id"},
            axis=1,
        )
        if "query" in run.columns:
            self._queries = run.drop_duplicates("query_id").set_index("query_id")["query"].rename("text")
            run = run.drop("query", axis=1)
        if "text" in run.columns:
            self._docs = run.set_index("doc_id")["text"].map(lambda x: GenericDoc("", x)).to_dict()
            run = run.drop("text", axis=1)
        if self.depth != -1:
            run = run[run["rank"] <= self.depth]
        dtypes = {"rank": np.int32}
        if "score" in run.columns:
            dtypes["score"] = np.float32
        run = run.astype(dtypes)
        return run

    def load_run(self) -> pd.DataFrame:

        suffix_load_map = {
            ".tsv": self.load_csv,
            ".run": self.load_csv,
            ".csv": self.load_csv,
            ".parquet": self.load_parquet,
            ".json": self.load_json,
            ".jsonl": self.load_json,
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
        if self._qrels is not None:
            return self._qrels
        if "relevance" in self.run:
            qrels = self.run[["query_id", "doc_id", "relevance"]].copy()
            if "iteration" in self.run:
                qrels["iteration"] = self.run["iteration"]
            else:
                qrels["iteration"] = "0"
            self.run = self.run.drop(["relevance", "iteration"], axis=1, errors="ignore")
            qrels = qrels.drop_duplicates(["query_id", "doc_id", "iteration"])
            qrels = qrels.set_index(["query_id", "doc_id", "iteration"]).unstack(level=-1)
            qrels = qrels.droplevel(0, axis=1)
            self._qrels = qrels
            return self._qrels
        return super().qrels

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> RankSample:
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
                .stack()
                .rename("relevance")
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
        super().__init__(tuples_dataset)
        super(IRDataset, self).__init__()
        self.targets = targets
        self.num_docs = num_docs

    def parse_sample(
        self, sample: ScoredDocTuple | GenericDocPair
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[float, ...] | None]:
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
                raise ValueError(f"invalid value for targets, got {self.targets}, " "expected one of (order, score)")
            targets = targets[: self.num_docs]
        else:
            raise ValueError("Invalid sample type.")
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)
        return doc_ids, docs, targets

    def __iter__(self) -> Iterator[RankSample]:
        for sample in self.ir_dataset.docpairs_iter():
            query_id = sample.query_id
            query = self.queries.loc[query_id]
            doc_ids, docs, targets = self.parse_sample(sample)
            if targets is not None:
                targets = torch.tensor(targets)
            yield RankSample(query_id, query, doc_ids, docs, targets)
