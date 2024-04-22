from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig

from ..bi_encoder.module import BiEncoderConfig, BiEncoderModule
from ..cross_encoder.module import CrossEncoderConfig, CrossEncoderModule
from ..tokenizer.tokenizer import BiEncoderTokenizer, CrossEncoderTokenizer
from .data import (
    BiEncoderTrainBatch,
    CrossEncoderTrainBatch,
    DocSample,
    IndexBatch,
    QuerySample,
    SearchBatch,
    TrainSample,
)
from .dataset import (
    DocDataset,
    DocDatasetConfig,
    QueryDataset,
    QueryDatasetConfig,
    RunDataset,
    RunDatasetConfig,
    TupleDatasetConfig,
    TuplesDataset,
)


class LightningIRDataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        config: BiEncoderConfig | CrossEncoderConfig | None = None,
        model: BiEncoderModule | CrossEncoderModule | None = None,
        num_workers: int = 0,
        train_batch_size: int | None = None,
        shuffle_train: bool = True,
        inference_batch_size: int | None = None,
        train_dataset: str | None = None,
        train_dataset_config: RunDatasetConfig | TupleDatasetConfig | None = None,
        inference_datasets: Sequence[str] | None = None,
        inference_dataset_config: (
            RunDatasetConfig
            | TupleDatasetConfig
            | QueryDatasetConfig
            | DocDatasetConfig
            | None
        ) = None,
    ) -> None:
        super().__init__()
        if model is not None:
            self.config = model.config
        elif config is not None and model_name_or_path is not None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            if config is not None:
                self.config = config.from_other(self.config)
                self.config.update(config.to_added_args_dict())
        else:
            raise ValueError(
                "Either a model or a model_name_or_path and config must be provided."
            )

        if isinstance(self.config, BiEncoderConfig):
            Tokenizer = BiEncoderTokenizer
        elif isinstance(self.config, CrossEncoderConfig):
            if isinstance(
                inference_dataset_config, (QueryDatasetConfig, DocDatasetConfig)
            ):
                raise ValueError(
                    "Running a cross-encoder model with a query or doc dataset is not "
                    "supported. Use a bi-encoder model instead."
                )
            Tokenizer = CrossEncoderTokenizer
        else:
            raise ValueError(
                f"LightningIRDataModule requires a BiEncoderConfig or "
                f"CrossEncoderConfig, received {self.config.__class__.__name__}."
            )
        self.tokenizer = Tokenizer.from_pretrained(
            model_name_or_path, **self.config.to_tokenizer_dict()
        )
        self.num_workers = num_workers

        self.train_batch_size = train_batch_size
        self.shuffle_train = shuffle_train
        self.inference_batch_size = inference_batch_size
        self.train_dataset = train_dataset
        self.inference_datasets = inference_datasets
        self.train_dataset_config = train_dataset_config
        self.inference_dataset_config = inference_dataset_config

    def setup_fit(self) -> None:
        if self.train_dataset is None or self.train_dataset_config is None:
            raise ValueError("A training dataset and config must be provided.")
        if isinstance(self.train_dataset_config, RunDatasetConfig):
            self._train_dataset = RunDataset(
                Path(self.train_dataset), self.train_dataset_config
            )
        elif isinstance(self.train_dataset_config, TupleDatasetConfig):
            self._train_dataset = TuplesDataset(
                self.train_dataset, self.train_dataset_config
            )
        else:
            raise ValueError(
                "Training DatasetConfig must be of type RunDatasetConfig or "
                "TupleDatasetConfig."
            )

    def setup_inference(self) -> None:
        if self.inference_datasets is None:
            return
        if self.inference_dataset_config is None:
            raise ValueError(
                "An inference DatasetConfig must be provided when "
                "providing inference datasets."
            )
        elif isinstance(self.inference_dataset_config, TupleDatasetConfig):
            self._inference_datasets = [
                TuplesDataset(dataset, self.inference_dataset_config)
                for dataset in self.inference_datasets
            ]
        elif isinstance(self.inference_dataset_config, RunDatasetConfig):
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

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.setup_fit()
        self.setup_inference()

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise ValueError("No training dataset found.")
        return DataLoader(
            self._train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=(
                False
                if isinstance(self._train_dataset, IterableDataset)
                else self.shuffle_train
            ),
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
            "query_id": {"extend": False, "tensorize": False},
            "query": {"extend": False, "tensorize": False},
            "doc_id": {"extend": False, "tensorize": False},
            "doc_ids": {"extend": False, "tensorize": False},
            "doc": {"extend": False, "tensorize": False},
            "docs": {"extend": True, "tensorize": False},
            "targets": {"extend": True, "tensorize": False},
            "qrels": {"extend": True, "tensorize": False},
        }
        for sample in samples:
            for field in sample._fields:
                extend = field_options[field]["extend"]
                tensorize = field_options[field]["tensorize"]
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
        queries = None
        if "querys" in aggregated:
            queries = aggregated["querys"]
            del kwargs["querys"]
        docs = None
        if "docs" in aggregated:
            docs = aggregated["docs"]
            del kwargs["docs"]
        encodings = self.tokenizer.tokenize(
            queries, docs, return_tensors="pt", padding=True, truncation=True
        )
        if not encodings:
            raise ValueError("No encodings were generated.")
        kwargs.update(encodings)
        if "targets" in aggregated:
            kwargs["targets"] = torch.stack(aggregated["targets"])
        return kwargs

    def _parse_batch(
        self, sample: TrainSample | QuerySample | DocSample, **kwargs
    ) -> BiEncoderTrainBatch | CrossEncoderTrainBatch | IndexBatch | SearchBatch:
        if isinstance(sample, TrainSample):
            if isinstance(self.config, BiEncoderConfig):
                return BiEncoderTrainBatch(**kwargs)
            elif isinstance(self.config, CrossEncoderConfig):
                return CrossEncoderTrainBatch(**kwargs)
            else:
                raise ValueError(
                    f"LightningIRDataModule requires a BiEncoderConfig or "
                    f"CrossEncoderConfig, received {self.config.__class__.__name__}."
                )
        if isinstance(sample, QuerySample):
            return SearchBatch(**kwargs)
        if isinstance(sample, DocSample):
            return IndexBatch(**kwargs)
        raise ValueError("Invalid dataset configuration.")

    def collate_fn(
        self, samples: Sequence[TrainSample | QuerySample | DocSample]
    ) -> BiEncoderTrainBatch | CrossEncoderTrainBatch | IndexBatch | SearchBatch:
        aggregated = self._aggregate_samples(samples)
        kwargs = self._clean_sample(aggregated)
        return self._parse_batch(samples[0], **kwargs)
