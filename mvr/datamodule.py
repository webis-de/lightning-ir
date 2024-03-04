import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, NamedTuple, Sequence
from itertools import islice

import ir_datasets
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.distributed import get_rank, get_world_size
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
)
from transformers import AutoConfig

from mvr.data import (
    DocSample,
    IndexBatch,
    QuerySample,
    SearchBatch,
    TrainBatch,
    TrainSample,
)
from mvr.mvr import MVRConfig, MVRTokenizer

DASHED_DATASET_MAP = {
    dataset.replace("/", "-"): dataset for dataset in ir_datasets.registry._registered
}
RUN_HEADER = ["query_id", "q0", "doc_id", "rank", "score", "system"]


class QueryDatasetConfig(NamedTuple):
    num_queries: int | None = None


class DocDatasetConfig(NamedTuple):
    num_docs: int | None = None


class RunDatasetConfig(NamedTuple):
    targets: Literal["relevance", "rank", "score"]
    depth: int
    sample_size: int
    sampling_strategy: Literal["single_relevant", "top"]


class TupleDatasetConfig(NamedTuple):
    num_docs: int | None


class DataParallelIterableDataset(IterableDataset):
    # https://github.com/Lightning-AI/pytorch-lightning/issues/15734
    def __init__(
        self, dataset: str, config: QueryDatasetConfig | DocDatasetConfig
    ) -> None:
        super().__init__()
        # TODO add support for multi-gpu and multi-worker inference; currently
        # doesn't work
        self.ir_dataset: ir_datasets.Dataset = ir_datasets.load(dataset)
        self.config = config
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        self.dataset_id = self.ir_dataset.dataset_id()
        self.docs_dataset_id = ir_datasets.docs_parent_id(self.dataset_id).replace(
            "/", "-"
        )

        try:
            world_size = get_world_size()
            process_rank = get_rank()
        except RuntimeError:
            world_size = 1
            process_rank = 0

        self.num_replicas = num_workers * world_size
        self.rank = process_rank * num_workers + worker_id
        if isinstance(config, QueryDatasetConfig):
            self._field = "queries"
            self._iterator = self.ir_dataset.queries_iter
            self._sample_cls = QuerySample
        elif isinstance(config, DocDatasetConfig):
            self._field = "docs"
            self._iterator = self.ir_dataset.docs_iter
            self._sample_cls = DocSample
        else:
            raise ValueError("Invalid dataset configuration.")

    def __len__(self) -> int:
        return (
            getattr(self.config, f"num_{self._field}")
            or getattr(self.ir_dataset, f"{self._field}_count")()
        )

    def __iter__(self) -> Iterator[QuerySample | DocSample]:
        start = self.rank
        stop = getattr(self.config, f"num_{self._field}") or None
        step = self.num_replicas
        for sample in islice(self._iterator(), start, stop, step):
            yield self._sample_cls.from_ir_dataset_sample(sample)


class QueryDataset(DataParallelIterableDataset):
    def __init__(self, query_dataset: str, config: QueryDatasetConfig) -> None:
        super().__init__(query_dataset, config)
        self.config: QueryDatasetConfig

    def __iter__(self) -> Iterator[QuerySample]:
        yield from super().__iter__()


class DocDataset(DataParallelIterableDataset):
    def __init__(self, doc_dataset: str, config: DocDatasetConfig) -> None:
        super().__init__(doc_dataset, config)
        self.config: DocDatasetConfig

    def __iter__(self) -> Iterator[DocSample]:
        yield from super().__iter__()


class IRDataset:
    def __init__(self, dataset: str) -> None:
        self.ir_dataset = ir_datasets.load(dataset)
        queries_iter = self.ir_dataset.queries_iter()
        self.queries = pd.DataFrame(queries_iter).set_index("query_id")["text"]
        self.docs = self.ir_dataset.docs_store()
        self.dataset_id = self.ir_dataset.dataset_id()
        self.docs_dataset_id = ir_datasets.docs_parent_id(self.dataset_id)


class RunDataset(IRDataset, Dataset):
    def __init__(
        self,
        run_dataset: Path,
        config: RunDatasetConfig,
    ) -> None:
        super().__init__(DASHED_DATASET_MAP[run_dataset.stem.split("__")[-1]])
        self.run = pd.read_csv(
            run_dataset,
            sep=r"\s+",
            header=None,
            names=RUN_HEADER,
            usecols=[0, 2, 3, 4],
            dtype={"query_id": str, "doc_id": str},
        )
        self.config = config
        self.depth = config.depth
        if self.depth != -1:
            self.run = self.run[self.run["rank"] <= config.depth]

        self.qrels = pd.DataFrame(self.ir_dataset.qrels_iter()).set_index(
            ["query_id", "doc_id"]
        )["relevance"]
        self.qrels = self.qrels.loc[
            pd.IndexSlice[self.run["query_id"].drop_duplicates(), :]
        ]
        self.run = self.run.merge(self.qrels, on=["query_id", "doc_id"], how="outer")
        self.run = self.run.sort_values(["query_id", "rank"])
        self.groups = self.run.groupby("query_id")
        self.query_ids = list(self.groups.groups.keys())

        if self.run["rank"].max() < config.depth:
            warnings.warn("Depth is greater than the maximum rank in the run file.")
        if config.sampling_strategy == "top" and config.sample_size > config.depth:
            warnings.warn(
                "Sample size is greater than depth and top sampling strategy is used. "
                "This can cause documents to be sampled that are not contained "
                "in the run file, but that are present in the qrels."
            )

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> TrainSample:
        query_id = str(self.query_ids[idx])
        group = self.groups.get_group(query_id).copy()
        query = self.queries[query_id]
        if self.config.sampling_strategy == "single_relevant":
            relevant = group.loc[group["relevance"] > 0].sample(1)
            non_relevant_bool = (
                group["relevance"].fillna(0).eq(0) & ~group["rank"].isna()
            )
            num_non_relevant = non_relevant_bool.sum()
            sample_non_relevant = min(self.config.sample_size - 1, num_non_relevant)
            non_relevant = group.loc[non_relevant_bool].sample(sample_non_relevant)
            group = pd.concat([relevant, non_relevant])
            relevance = tuple([1] + [0] * sample_non_relevant)
        else:
            relevance = tuple(group["relevance"].fillna(0))
            group = group.head(self.config.sample_size)

        doc_ids = tuple(group["doc_id"])
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)
        targets = tuple(
            group.set_index("doc_id").loc[list(doc_ids), self.config.targets].fillna(0)
        )
        return TrainSample(query_id, query, doc_ids, docs, targets, relevance)


class TuplesDataset(IRDataset, IterableDataset):
    def __init__(self, tuples_dataset: str, config: TupleDatasetConfig) -> None:
        super().__init__(tuples_dataset)
        if self.queries is None:
            raise ValueError("Queries are required for run datasets.")
        self.config = config

    def __iter__(self) -> Iterator[TrainSample]:
        for sample in self.ir_dataset.docpairs_iter():
            query_id = sample.query_id
            query = self.queries.loc[query_id]

            doc_ids = sample.doc_ids[: self.config.num_docs]
            docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)

            scores = (
                sample.scores
                if sample.scores is not None
                else tuple([1.0] + [0.0] * sample.num_docs)
            )
            scores = scores[: self.config.num_docs]

            yield TrainSample(
                query_id,
                query,
                doc_ids,
                docs,
                scores,
            )


class MVRDataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str | Path,
        config: MVRConfig | None,
        num_workers: int = 0,
        train_batch_size: int | None = None,
        inference_batch_size: int | None = None,
        train_dataset: str | None = None,
        train_dataset_config: RunDatasetConfig | TupleDatasetConfig | None = None,
        inference_datasets: Sequence[str] | None = None,
        inference_dataset_config: (
            RunDatasetConfig | QueryDatasetConfig | DocDatasetConfig | None
        ) = None,
    ) -> None:
        super().__init__()
        self.config = MVRConfig.from_other(
            AutoConfig.from_pretrained(model_name_or_path)
        )
        if config is not None:
            self.config.update(config.to_mvr_dict())
        self.tokenizer = MVRTokenizer.from_pretrained(
            model_name_or_path, **self.config.to_tokenizer_dict()
        )
        self.num_workers = num_workers

        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.train_dataset = train_dataset
        self.inference_datasets = inference_datasets
        self.train_dataset_config = train_dataset_config
        self.inference_dataset_config = inference_dataset_config

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.train_dataset is None or self.train_dataset_config is None:
                raise ValueError("A training dataset and config must be provided.")
            if isinstance(self.train_dataset_config, RunDatasetConfig):
                self._train_dataset = RunDataset(
                    Path(self.train_dataset), self.train_dataset_config
                )
            else:
                self._train_dataset = TuplesDataset(
                    self.train_dataset, self.train_dataset_config
                )
        if self.inference_datasets is not None:
            if self.inference_dataset_config is None:
                raise ValueError(
                    "An inference DatasetConfig must be provided when "
                    "providing a inference datasets."
                )
            if isinstance(self.inference_dataset_config, RunDatasetConfig):
                if self.inference_dataset_config.sampling_strategy == "single_relevant":
                    raise ValueError(
                        "Inference RunDatasetConfig cannot use the single_relevant "
                        "sampling strategy."
                    )
                self._inference_datasets = [
                    RunDataset(Path(dataset), self.inference_dataset_config)
                    for dataset in self.inference_datasets
                ]
            elif isinstance(self.inference_dataset_config, QueryDatasetConfig):
                self._inference_datasets = [
                    QueryDataset(dataset, self.inference_dataset_config)
                    for dataset in self.inference_datasets
                ]
            elif isinstance(self.inference_dataset_config, DocDatasetConfig):
                self._inference_datasets = [
                    DocDataset(dataset, self.inference_dataset_config)
                    for dataset in self.inference_datasets
                ]
            else:
                raise ValueError(
                    "Inference DatasetConfig must be of type RunDatasetConfig, "
                    "QueryDatasetConfig, or DocDatasetConfig."
                )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise ValueError("No training dataset found.")
        return DataLoader(
            self._train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return self.inference_dataloader()

    def predict_dataloader(self) -> List[DataLoader]:
        return self.inference_dataloader()

    def inference_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                dataset,
                batch_size=self.inference_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for dataset in self._inference_datasets
        ]

    def _aggregate_samples(
        self, samples: Sequence[TrainSample | QuerySample | DocSample]
    ) -> Dict[str, Any]:
        aggregated = defaultdict(list)
        field_options = {
            "query_id": (False, False),
            "query": (False, False),
            "doc_id": (False, False),
            "doc_ids": (False, False),
            "doc": (False, False),
            "docs": (True, False),
            "targets": (True, False),
            "relevances": (False, True),
        }
        for sample in samples:
            for field in sample._fields:
                extend, tensorize = field_options[field]
                key = field if field.endswith("s") else f"{field}s"
                value = getattr(sample, field)
                if value is None:
                    continue
                if tensorize:
                    value = torch.tensor(value)
                if extend:
                    aggregated[key].extend(value)
                else:
                    aggregated[key].append(value)
        return aggregated

    def _clean_sample(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(aggregated)
        if "querys" in aggregated:
            kwargs["query_encoding"] = self.tokenizer.tokenize_queries(
                aggregated["querys"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.query_length,
            )
            del kwargs["querys"]
        if "docs" in aggregated:
            kwargs["doc_encoding"] = self.tokenizer.tokenize_docs(
                aggregated["docs"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.doc_length,
            )
            del kwargs["docs"]
        if "targets" in aggregated:
            kwargs["targets"] = torch.tensor(aggregated["targets"])
        if "relevances" in aggregated:
            kwargs["relevances"] = torch.nn.utils.rnn.pad_sequence(
                aggregated["relevances"], batch_first=True
            )
        return kwargs

    def _parse_batch(self, **kwargs) -> TrainBatch | IndexBatch | SearchBatch:
        if self.train_dataset_config is not None:
            return TrainBatch(**kwargs)
        if self.inference_dataset_config is not None:
            if isinstance(self.inference_dataset_config, QueryDatasetConfig):
                return SearchBatch(**kwargs)
            if isinstance(self.inference_dataset_config, DocDatasetConfig):
                return IndexBatch(**kwargs)
        raise ValueError("Invalid dataset configuration.")

    def collate_fn(
        self, samples: Sequence[TrainSample | QuerySample | DocSample]
    ) -> TrainBatch | IndexBatch | SearchBatch:
        aggregated = self._aggregate_samples(samples)
        kwargs = self._clean_sample(aggregated)
        return self._parse_batch(**kwargs)
