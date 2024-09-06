from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from ..base.config import LightningIRConfig
from ..base.tokenizer import LightningIRTokenizer
from .data import IndexBatch, RankBatch, SearchBatch, TrainBatch
from .dataset import DocDataset, DocSample, QueryDataset, QuerySample, RankSample, RunDataset, TupleDataset

if TYPE_CHECKING:
    from ..base import LightningIRModule


class LightningIRDataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        config: LightningIRConfig | None = None,
        module: LightningIRModule | None = None,
        num_workers: int = 0,
        train_batch_size: int | None = None,
        shuffle_train: bool = True,
        inference_batch_size: int | None = None,
        train_dataset: RunDataset | TupleDataset | None = None,
        inference_datasets: Sequence[RunDataset | TupleDataset | QueryDataset | DocDataset] | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            self.config = config
        elif module is not None:
            self.config = module.config
        elif model_name_or_path is not None:
            self.config = LightningIRConfig.from_pretrained(model_name_or_path)
        else:
            raise ValueError("Either module, config, or model_name_or_path must be provided.")

        if model_name_or_path is None:
            model_name_or_path = self.config.name_or_path
        self.tokenizer = LightningIRTokenizer.from_pretrained(model_name_or_path, config=self.config)
        self.num_workers = num_workers

        self.train_batch_size = train_batch_size
        self.shuffle_train = shuffle_train
        self.inference_batch_size = inference_batch_size
        self.train_dataset = train_dataset
        self.inference_datasets = inference_datasets

    def setup_inference(self, stage: Literal["validate", "test"]) -> None:
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

    def setup(self, stage: Literal["fit", "validate", "test"]) -> None:
        if stage == "fit":
            if self.train_dataset is None:
                raise ValueError("A training dataset and config must be provided.")
        if stage == "fit":
            stage = "validate"
        self.setup_inference(stage)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("No training dataset found.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=(False if isinstance(self.train_dataset, IterableDataset) else self.shuffle_train),
            prefetch_factor=16 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return self.inference_dataloader()

    def test_dataloader(self) -> List[DataLoader]:
        return self.inference_dataloader()

    def inference_dataloader(self) -> List[DataLoader]:
        inference_datasets = self.inference_datasets or []
        return [
            DataLoader(
                dataset,
                batch_size=self.inference_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                prefetch_factor=16 if self.num_workers > 0 else None,
            )
            for dataset in inference_datasets
        ]

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

    def collate_fn(
        self,
        samples: Sequence[RankSample | QuerySample | DocSample] | RankSample | QuerySample | DocSample,
    ) -> TrainBatch | RankBatch | IndexBatch | SearchBatch:
        if isinstance(samples, (RankSample, QuerySample, DocSample)):
            samples = [samples]
        aggregated = self._aggregate_samples(samples)
        kwargs = self._clean_sample(aggregated)
        return self._parse_batch(samples[0], **kwargs)
