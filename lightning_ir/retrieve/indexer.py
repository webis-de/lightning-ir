from __future__ import annotations

import array
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..bi_encoder import BiEncoderConfig, BiEncoderOutput
    from ..data import IndexBatch


class Indexer(ABC):
    def __init__(
        self,
        index_dir: Path,
        index_config: IndexConfig,
        bi_encoder_config: BiEncoderConfig,
        verbose: bool = False,
    ) -> None:
        self.index_dir = index_dir
        self.index_config = index_config
        self.bi_encoder_config = bi_encoder_config
        self.doc_ids = []
        self.doc_lengths = array.array("I")
        self.num_embeddings = 0
        self.num_docs = 0
        self.verbose = verbose

    @abstractmethod
    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None: ...

    def save(self) -> None:
        self.index_config.save(self.index_dir)
        (self.index_dir / "doc_ids.txt").write_text("\n".join(self.doc_ids))
        doc_lengths = torch.tensor(self.doc_lengths)
        torch.save(doc_lengths, self.index_dir / "doc_lengths.pt")


class IndexConfig:
    indexer_class = Indexer

    @classmethod
    def from_pretrained(cls, index_dir: Path) -> "IndexConfig":
        with open(index_dir / "config.json", "r") as f:
            data = json.load(f)
            if data["index_type"] != cls.__name__:
                raise ValueError(f"Expected index_type {cls.__name__}, got {data['index_type']}")
            data.pop("index_type", None)
            data.pop("index_dir", None)
            return cls(**data)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "config.json", "w") as f:
            data = self.__dict__.copy()
            data["index_dir"] = str(index_dir)
            data["index_type"] = self.__class__.__name__
            json.dump(data, f)
