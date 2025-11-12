"""Base indexer class and configuration for retrieval tasks."""

from __future__ import annotations

import array
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderModule, BiEncoderOutput
    from ...data import IndexBatch


class Indexer(ABC):
    """Base class for indexers that create and manage indices for retrieval tasks."""

    def __init__(
        self,
        index_dir: Path,
        index_config: IndexConfig,
        module: BiEncoderModule,
        verbose: bool = False,
    ) -> None:
        """Initialize the Indexer.

        Args:
            index_dir (Path): Directory where the index will be stored.
            index_config (IndexConfig): Configuration for the index.
            module (BiEncoderModule): The bi-encoder module used for encoding documents.
            verbose (bool): Whether to print verbose output. Defaults to False.
        """
        self.index_dir = index_dir
        self.index_config = index_config
        self.module = module
        self.doc_ids: list[str] = []
        self.doc_lengths = array.array("I")
        self.num_embeddings = 0
        self.num_docs = 0
        self.verbose = verbose

    @abstractmethod
    def add(self, index_batch: IndexBatch, output: BiEncoderOutput) -> None:
        """Add a batch of documents to the index.

        Args:
            index_batch (IndexBatch): The batch of documents to add.
            output (BiEncoderOutput): The output from the bi-encoder module containing document embeddings.
        """
        ...

    def save(self) -> None:
        """Save the index configuration and document IDs to the index directory."""
        self.index_config.save(self.index_dir)
        (self.index_dir / "doc_ids.txt").write_text("\n".join(self.doc_ids))
        doc_lengths = torch.frombuffer(self.doc_lengths, dtype=torch.int32)
        torch.save(doc_lengths, self.index_dir / "doc_lengths.pt")


class IndexConfig:
    """Configuration class for indexers that defines the index type and other parameters."""

    indexer_class: type[Indexer]
    SUPPORTED_MODELS: set[str]

    @classmethod
    def from_pretrained(cls, index_dir: Path | str) -> IndexConfig:
        """Load the index configuration from a directory.

        Args:
            index_dir (Path | str): Path to the directory containing the index configuration.
        Returns:
            IndexConfig: An instance of the index configuration class.
        Raises:
            ValueError: If the index type in the configuration does not match the expected class name.
        """
        index_dir = Path(index_dir)
        with open(index_dir / "config.json") as f:
            data = json.load(f)
            if data["index_type"] != cls.__name__:
                raise ValueError(f"Expected index_type {cls.__name__}, got {data['index_type']}")
            data.pop("index_type", None)
            data.pop("index_dir", None)
            return cls(**data)

    def save(self, index_dir: Path) -> None:
        """Save the index configuration to a directory.

        Args:
            index_dir (Path): The directory to save the index configuration.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "config.json", "w") as f:
            data = self.__dict__.copy()
            data["index_dir"] = str(index_dir)
            data["index_type"] = self.__class__.__name__
            json.dump(data, f)

    def to_dict(self) -> dict:
        """Convert the index configuration to a dictionary.

        Returns:
            dict: A dictionary representation of the index configuration.
        """
        return self.__dict__.copy()
