from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig

from ..bi_encoder.module import BiEncoderConfig, BiEncoderModule
from ..cross_encoder.module import CrossEncoderConfig, CrossEncoderModule
from .data import (
    BiEncoderRunBatch,
    CrossEncoderRunBatch,
    DocSample,
    IndexBatch,
    QuerySample,
    RunSample,
    SearchBatch,
)
from .dataset import (
    DocDataset,
    QueryDataset,
    RunDataset,
    TupleDataset,
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
        train_dataset: RunDataset | TupleDataset | None = None,
        inference_datasets: (
            Sequence[RunDataset | TupleDataset | QueryDataset | DocDataset] | None
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

        self.tokenizer = self.config.Tokenizer.from_pretrained(
            model_name_or_path, **self.config.to_tokenizer_dict()
        )
        self.num_workers = num_workers

        self.train_batch_size = train_batch_size
        self.shuffle_train = shuffle_train
        self.inference_batch_size = inference_batch_size
        self.train_dataset = train_dataset
        self.inference_datasets = inference_datasets

    def setup_fit(self) -> None:
        if self.train_dataset is None:
            raise ValueError("A training dataset and config must be provided.")
        self.train_dataset.setup("fit")

    def setup_inference(self, stage: Literal["validate", "predict"]) -> None:
        if self.inference_datasets is None:
            return
        for inference_dataset in self.inference_datasets:
            if isinstance(inference_dataset, TupleDataset):
                if stage == "predict":
                    raise ValueError(
                        "Prediction cannot be performed with TupleDataset."
                    )
            elif isinstance(inference_dataset, RunDataset):
                if inference_dataset.sampling_strategy == "single_relevant":
                    raise ValueError(
                        "Inference RunDataset cannot use the single_relevant "
                        "sampling strategy."
                    )
            elif isinstance(inference_dataset, (QueryDataset, DocDataset)):
                pass
            else:
                raise ValueError(
                    "Inference Dataset must be of type RunDataset, "
                    "TupleDataset, QueryDataset, or DocDataset."
                )
            inference_dataset.setup(stage)

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        if stage == "fit":
            self.setup_fit()
        if stage == "fit":
            stage = "validate"
        if stage == "test":
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
            shuffle=(
                False
                if isinstance(self.train_dataset, IterableDataset)
                else self.shuffle_train
            ),
        )

    def val_dataloader(self) -> List[DataLoader]:
        return self.inference_dataloader()

    def predict_dataloader(self) -> List[DataLoader]:
        return self.inference_dataloader()

    def inference_dataloader(self) -> List[DataLoader]:
        inference_datasets = self.inference_datasets or []
        return [
            DataLoader(
                dataset,
                batch_size=self.inference_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for dataset in inference_datasets
        ]

    def _aggregate_samples(
        self, samples: Sequence[RunSample | QuerySample | DocSample]
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
        num_docs = None
        if "doc_ids" in aggregated:
            num_docs = [len(doc_ids) for doc_ids in aggregated["doc_ids"]]
        encodings = self.tokenizer.tokenize(
            queries,
            docs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            num_docs=num_docs,
        )
        if not encodings:
            raise ValueError("No encodings were generated.")
        kwargs.update(encodings)
        if "targets" in aggregated:
            kwargs["targets"] = torch.stack(aggregated["targets"])
        return kwargs

    def _parse_batch(
        self, sample: RunSample | QuerySample | DocSample, **kwargs
    ) -> BiEncoderRunBatch | CrossEncoderRunBatch | IndexBatch | SearchBatch:
        if isinstance(sample, RunSample):
            if isinstance(self.config, BiEncoderConfig):
                return BiEncoderRunBatch(**kwargs)
            elif isinstance(self.config, CrossEncoderConfig):
                return CrossEncoderRunBatch(**kwargs)
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
        self,
        samples: Sequence[
            RunSample | QuerySample | DocSample | RunSample | QuerySample | DocSample
        ],
    ) -> BiEncoderRunBatch | CrossEncoderRunBatch | IndexBatch | SearchBatch:
        if isinstance(samples, (RunSample, QuerySample, DocSample)):
            samples = [samples]
        aggregated = self._aggregate_samples(samples)
        kwargs = self._clean_sample(aggregated)
        return self._parse_batch(samples[0], **kwargs)
