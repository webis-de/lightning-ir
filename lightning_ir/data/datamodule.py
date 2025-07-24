"""
DataModule for Lightning IR that handles batching and collation of data samples.

This module defines the LightningIRDataModule class that handles batching and collation of data samples for training and
inference in Lightning IR.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Literal, Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from .data import IndexBatch, RankBatch, SearchBatch, TrainBatch
from .dataset import (
    DocDataset,
    DocSample,
    QueryDataset,
    QuerySample,
    RankSample,
    RunDataset,
    TupleDataset,
    _DummyIterableDataset,
)


class LightningIRDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: RunDataset | TupleDataset | None = None,
        train_batch_size: int | None = None,
        shuffle_train: bool = True,
        inference_datasets: Sequence[RunDataset | TupleDataset | QueryDataset | DocDataset] | None = None,
        inference_batch_size: int | None = None,
        num_workers: int = 0,
    ) -> None:
        """Initializes a new Lightning IR DataModule.

        Args:
            train_dataset (RunDataset | TupleDataset | None): A training dataset. Defaults to None.
            train_batch_size (int | None): Batch size to use for training. Defaults to None.
            shuffle_train (bool): Whether to shuffle the training data. Defaults to True.
            inference_datasets (Sequence[RunDataset | TupleDataset | QueryDataset | DocDataset] | None): List of
                datasets to use for inference (indexing, searching, and re-ranking). Defaults to None.
            inference_batch_size (int | None): Batch size to use for inference. Defaults to None.
            num_workers (int): Number of workers for loading data in parallel. Defaults to 0.
        """
        super().__init__()
        self.num_workers = num_workers

        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.shuffle_train = shuffle_train
        self.inference_datasets = None if inference_datasets is None else list(inference_datasets)
        self.inference_batch_size = inference_batch_size

        if (self.train_batch_size is not None) != (self.train_dataset is not None):
            raise ValueError("Both train_batch_size and train_dataset must be provided.")
        if (self.inference_batch_size is not None) != (self.inference_datasets is not None):
            raise ValueError("Both inference_batch_size and inference_dataset must be provided.")

    def _setup_inference(self, stage: Literal["validate", "test"]) -> None:
        if self.inference_datasets is None:
            return
        for inference_dataset in self.inference_datasets:
            if isinstance(inference_dataset, TupleDataset):
                if stage == "test":
                    raise ValueError("Prediction cannot be performed with TupleDataset.")
            elif isinstance(inference_dataset, RunDataset):
                if inference_dataset.sampling_strategy == "single_relevant":
                    raise ValueError("Inference RunDataset cannot use the single_relevant sampling strategy.")
            elif isinstance(inference_dataset, (QueryDataset, DocDataset)):
                pass
            else:
                raise ValueError(
                    "Inference Dataset must be of type RunDataset, TupleDataset, QueryDataset, or DocDataset."
                )

    def prepare_data(self) -> None:
        """Downloads the data using ir_datasets if needed."""
        if self.train_dataset is not None:
            self.train_dataset.prepare_data()
        if self.inference_datasets is not None:
            for inference_dataset in self.inference_datasets:
                inference_dataset.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test"]) -> None:
        """Sets up the data module for a given stage.

        Args:
            stage (Literal["fit", "validate", "test"]): Stage to set up the data module for.
        Raises:
            ValueError: If the stage is `fit` and no training dataset is provided.
        """
        if stage == "fit":
            if self.train_dataset is None:
                raise ValueError("A training dataset and config must be provided.")
        if stage == "fit":
            stage = "validate"
        self._setup_inference(stage)

    def train_dataloader(self) -> DataLoader:
        """Returns a dataloader for training.

        Returns:
            DataLoader: Dataloader for training.
        Raises:
            ValueError: If no training dataset is found.
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset found.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            shuffle=(False if isinstance(self.train_dataset, IterableDataset) else self.shuffle_train),
            prefetch_factor=16 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> List[DataLoader]:
        """Returns a list of dataloaders for validation.

        Returns:
            List[DataLoader]: Dataloaders for validation.
        """
        return self.inference_dataloader()

    def test_dataloader(self) -> List[DataLoader]:
        """Returns a list of dataloaders for testing.

        Returns:
            List[DataLoader]: Dataloaders for testing.
        """
        return self.inference_dataloader()

    def predict_dataloader(self) -> Any:
        """Returns a list of dataloaders for predicting.

        Returns:
            List[DataLoader]: Dataloaders for predicting.
        """
        return self.inference_dataloader()

    def inference_dataloader(self) -> List[DataLoader]:
        """Returns a list of dataloaders for inference (validation, testing, or predicting).

        Returns:
            List[DataLoader]: Dataloaders for inference.
        """
        inference_datasets = self.inference_datasets or []
        dataloaders = [
            DataLoader(
                dataset,
                batch_size=self.inference_batch_size,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
                prefetch_factor=16 if self.num_workers > 0 else None,
            )
            for dataset in inference_datasets
            if not dataset._SKIP
        ]
        if not dataloaders:
            dataloaders = [DataLoader(_DummyIterableDataset())]
        return dataloaders

    def _aggregate_samples(self, samples: Sequence[RankSample | QuerySample | DocSample]) -> Dict[str, Any]:
        aggregated = defaultdict(list)
        field_options = {
            "query_id": {"extend": False},
            "query": {"extend": False},
            "doc_id": {"extend": False},
            "doc_ids": {"extend": False},
            "doc": {"extend": False},
            "docs": {"extend": False},
            "targets": {"extend": True},
            "qrels": {"extend": True},
        }
        for sample in samples:
            for field in sample.__dict__:
                extend = field_options[field]["extend"]
                key = field if field.endswith("s") else f"{field}s"
                value = getattr(sample, field)
                if value is None:
                    continue
                if extend:
                    aggregated[key].extend(value)
                else:
                    aggregated[key].append(value)
        return aggregated

    def _clean_sample(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(aggregated)
        if "querys" in kwargs:
            kwargs["queries"] = kwargs["querys"]
            del kwargs["querys"]
        if "targets" in kwargs:
            kwargs["targets"] = torch.stack(kwargs["targets"])
        return kwargs

    def _parse_batch(
        self, sample: RankSample | QuerySample | DocSample, **kwargs
    ) -> RankBatch | TrainBatch | IndexBatch | SearchBatch:
        if isinstance(sample, RankSample):
            if "targets" in kwargs:
                return TrainBatch(**kwargs)
            else:
                return RankBatch(**kwargs)
        if isinstance(sample, QuerySample):
            return SearchBatch(**kwargs)
        if isinstance(sample, DocSample):
            return IndexBatch(**kwargs)
        raise ValueError("Invalid dataset configuration.")

    def _collate_fn(
        self,
        samples: Sequence[RankSample | QuerySample | DocSample] | RankSample | QuerySample | DocSample,
    ) -> TrainBatch | RankBatch | IndexBatch | SearchBatch:
        if isinstance(samples, (RankSample, QuerySample, DocSample)):
            samples = [samples]
        aggregated = self._aggregate_samples(samples)
        kwargs = self._clean_sample(aggregated)
        return self._parse_batch(samples[0], **kwargs)
