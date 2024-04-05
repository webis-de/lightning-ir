import itertools
import math
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback, TQDMProgressBar

from .data import IndexBatch, SearchBatch
from .datamodule import RUN_HEADER, DocDataset, QueryDataset
from .indexer import IVFPQIndexConfig, IVFPQIndexer
from .module import MVRModule
from .searcher import SearchConfig, Searcher


def format_large_number(number: float) -> str:
    suffixes = ["", "K", "M", "B", "T"]
    suffix_index = 0

    while number >= 1000 and suffix_index < len(suffixes) - 1:
        number /= 1000.0
        suffix_index += 1

    formatted_number = "{:.2f}".format(number)

    suffix = suffixes[suffix_index]
    if suffix:
        formatted_number += f" {suffix}"
    return formatted_number


class IndexCallback(Callback):
    def __init__(
        self,
        index_dir: Path | None,
        num_train_tokens: int | None = None,
        num_centroids: int | None = None,
        num_subquantizers: int = 16,
        n_bits: int = 4,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.index_dir = index_dir
        self.num_train_tokens = num_train_tokens
        self.num_centroids = num_centroids
        self.num_subquantizers = num_subquantizers
        self.n_bits = n_bits
        self.verbose = verbose
        self.indexer: IVFPQIndexer

    def setup(self, trainer: Trainer, pl_module: MVRModule, stage: str) -> None:
        if stage != "predict":
            raise ValueError("IndexCallback can only be used in predict stage")

    def on_predict_start(self, trainer: Trainer, pl_module: MVRModule) -> None:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        datasets = [dataloader.dataset for dataloader in dataloaders]
        if not all(isinstance(dataset, DocDataset) for dataset in datasets):
            raise ValueError("Expected DocDatasets for indexing")

    def get_index_path(self, pl_module: MVRModule, dataset: DocDataset) -> Path:
        index_dir = self.index_dir
        if index_dir is None:
            default_index_dir = Path(pl_module.config.name_or_path)
            if default_index_dir.exists():
                index_dir = default_index_dir / "indexes"
            else:
                raise ValueError(
                    "No index_path provided and model_name_or_path is not a path"
                )
        index_path = index_dir / dataset.docs_dataset_id
        return index_path

    def get_indexer(
        self, trainer: Trainer, pl_module: MVRModule, dataset_idx: int
    ) -> IVFPQIndexer:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        dataset = dataloaders[dataset_idx].dataset

        index_path = self.get_index_path(pl_module, dataset)

        num_docs = dataset.ir_dataset.docs_count()
        approx_num_tokens = int(
            (
                sum(
                    len(doc.default_text().split())
                    for _, doc in zip(range(100), dataset.ir_dataset.docs_iter())
                )
                / 100
            )
            * num_docs
        )
        # default faiss values
        # https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/Clustering.h#L43
        max_points_per_centroid = 256

        num_centroids = self.num_centroids
        num_train_tokens = self.num_train_tokens
        # max 2^18 * max_points_per_centroid training tokens
        approx_num_tokens = int(
            min(2**18 * max_points_per_centroid, num_train_tokens or approx_num_tokens)
        )
        if num_centroids is None:
            num_centroids = 2 ** math.floor(
                math.log2(approx_num_tokens / max_points_per_centroid)
            )
        if num_train_tokens is None:
            num_train_tokens = approx_num_tokens

        config = IVFPQIndexConfig(
            index_path=index_path,
            num_train_tokens=num_train_tokens,
            num_centroids=num_centroids,
            num_subquantizers=self.num_subquantizers,
            n_bits=self.n_bits,
        )
        indexer = IVFPQIndexer(config, pl_module.config, self.verbose)
        return indexer

    def log_to_pg(self, info: Dict[str, Any], trainer: Trainer):
        pg_callback = trainer.progress_bar_callback
        if pg_callback is None or not isinstance(pg_callback, TQDMProgressBar):
            return
        pg = pg_callback.predict_progress_bar
        info = {k: format_large_number(v) for k, v in info.items()}
        if pg is not None:
            pg.set_postfix(info)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: MVRModule,
        outputs: Any,
        batch: IndexBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if batch_idx == 0:
            if hasattr(self, "indexer"):
                self.indexer.save()
            self.indexer = self.get_indexer(trainer, pl_module, dataloader_idx)

        doc_id_length = max(2, max(len(doc_id) for doc_id in batch.doc_ids))
        encoded_doc_ids = torch.ByteTensor(
            list(
                bytes(doc_id.rjust(doc_id_length), "utf32") for doc_id in batch.doc_ids
            )
        )
        outputs = pl_module.all_gather(outputs)
        encoded_doc_ids = pl_module.all_gather(encoded_doc_ids)
        attention_mask = batch.doc_encoding.attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(batch.doc_encoding.input_ids.shape)
        scoring_mask = pl_module.model.scoring_function.doc_scoring_mask(
            batch.doc_encoding.input_ids, attention_mask
        )
        scoring_mask = pl_module.all_gather(scoring_mask)
        if not trainer.is_global_zero:
            return
        outputs = outputs.view(-1, *outputs.shape[-2:])

        outputs = outputs.view(-1, pl_module.config.embedding_dim)
        embeddings = outputs[scoring_mask.bool().view(-1)]
        doc_ids = [bytes(doc_id).decode("utf32").strip() for doc_id in encoded_doc_ids]
        doc_lengths = scoring_mask.sum(-1)

        self.indexer.add(embeddings, doc_ids, doc_lengths)
        self.log_to_pg(
            {
                "num_docs": self.indexer.num_docs,
                "num_embeddings": self.indexer.num_embeddings,
            },
            trainer,
        )

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.indexer.save()


class SearchCallback(BasePredictionWriter):
    def __init__(
        self,
        save_dir: Path | None = None,
        index_path: Path | None = None,
        k: int = 100,
        candidate_k: int = 1000,
        imputation_strategy: Literal["min", "gather"] | None = None,
        n_probe: int = 1,
    ) -> None:
        super().__init__()
        if imputation_strategy is None:
            raise ValueError("imputation_strategy must be set")
        self.save_dir = save_dir
        self.index_path = index_path
        self.k = k
        self.candidate_k = candidate_k
        self.imputation_strategy = imputation_strategy
        self.n_probe = n_probe
        self.config: SearchConfig
        self.searcher: Searcher

    def setup(self, trainer: Trainer, pl_module: MVRModule, stage: str) -> None:
        if stage != "predict":
            raise ValueError("SearchCallback can only be used in predict stage")

    def on_predict_start(self, trainer: Trainer, pl_module: MVRModule) -> None:
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        datasets = [dataloader.dataset for dataloader in dataloaders]
        if not all(isinstance(dataset, QueryDataset) for dataset in datasets):
            raise ValueError("Expected QueryDatasets for searching")
        docs_dataset_id = [dataset.docs_dataset_id for dataset in datasets]
        if len(set(docs_dataset_id)) != 1:
            raise ValueError("All QueryDatasets must have the same docs_dataset_id")
        docs_dataset_id = docs_dataset_id[0]

        index_path = self.index_path
        if index_path is None:
            if Path(pl_module.config.name_or_path).exists():
                index_dir = Path(pl_module.config.name_or_path) / "indexes"
                index_path = index_dir / docs_dataset_id
                if not index_path.exists():
                    raise ValueError(f"No index found at {index_path}")
            else:
                raise ValueError(
                    "No index_path provided and model_name_or_path is not a path"
                )
        self.config = SearchConfig(
            index_path,
            self.k,
            self.candidate_k,
            self.imputation_strategy,
            self.n_probe,
        )
        self.searcher = Searcher(self.config, pl_module.config)
        if self.save_dir is None:
            default_save_dir = Path(pl_module.config.name_or_path)
            if default_save_dir.exists():
                self.save_dir = default_save_dir / "runs"
                print(f"Using default save_dir {self.save_dir}")
            else:
                raise ValueError(
                    "No index_path provided and model_name_or_path is not a path"
                )

    def get_run_path(self, trainer: Trainer, dataset_idx: int) -> Path:
        dataloaders = trainer.predict_dataloaders
        if self.save_dir is None:
            raise ValueError("No save_dir found; call setup before using this method")
        if dataloaders is None:
            raise ValueError("No predict_dataloaders found")
        dataset = dataloaders[dataset_idx].dataset
        if not isinstance(dataset, QueryDataset):
            raise ValueError("Expected a QueryDataset for searching")
        dataset_id = dataset.dataset_id.replace("/", "-")
        run_file_path = self.save_dir / f"{dataset_id}.run"
        return run_file_path

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: MVRModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: SearchBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        query_embeddings = pl_module.all_gather(prediction)
        query_attention_mask = pl_module.all_gather(
            batch.query_encoding.attention_mask.bool()
        )
        if not trainer.is_global_zero:
            return

        scores, doc_ids, num_docs = self.searcher.search(
            query_embeddings, query_attention_mask
        )
        scores = scores.cpu().numpy()

        query_ids = list(
            itertools.chain.from_iterable(
                itertools.repeat(query_id, num)
                for query_id, num in zip(batch.query_ids, num_docs)
            )
        )
        run_df = pd.DataFrame(
            zip(query_ids, doc_ids, scores), columns=["query_id", "doc_id", "score"]
        )
        run_df = run_df.sort_values(["query_id", "score"], ascending=[True, False])
        run_df["rank"] = (
            run_df.groupby("query_id")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        run_df["q0"] = 0
        run_df["system"] = pl_module.config.model_type
        run_df = run_df[RUN_HEADER]

        run_file_path = self.get_run_path(trainer, dataloader_idx)
        run_file_path.parent.mkdir(exist_ok=True)
        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"

        run_df.to_csv(run_file_path, header=False, index=False, sep="\t", mode=mode)
