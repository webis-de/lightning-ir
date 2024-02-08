from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from tide.datamodule import RUN_HEADER, MVRDataModule
from tide.mvr import MVRModule


class PredictionWriter(BasePredictionWriter):
    def __init__(self, overwrite: bool = False) -> None:
        super().__init__("batch")
        self.overwrite = overwrite

    def get_run_path(
        self, trainer: Trainer, pl_module: LightningModule, dataset_idx: int
    ) -> Path:
        if not hasattr(trainer, "ckpt_path") or trainer.ckpt_path is None:
            raise ValueError("ckpt_path is not set")
        if not hasattr(trainer, "datamodule") or trainer.datamodule is None:
            raise ValueError("datamodule is not set")
        ckpt_path = Path(trainer.ckpt_path)
        datamodule: MVRDataModule = trainer.datamodule
        if (
            datamodule.inference_datasets is None
            or datamodule.inference_run_config is None
        ):
            raise ValueError("inference datasets are not set")
        inference_run_path = Path(datamodule.inference_datasets[dataset_idx])
        dataset_id = inference_run_path.stem
        first_stage = inference_run_path.parent.name
        filename = (
            f"{first_stage}_{datamodule.inference_run_config.depth}_{dataset_id}.run"
        )
        run_file_path = ckpt_path.parent.parent / "runs" / filename
        return run_file_path

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: MVRModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        run_file_path = self.get_run_path(trainer, pl_module, dataloader_idx)
        doc_ids = batch["doc_ids"]
        query_ids = batch["query_id"]
        scores = [float(logit.item()) for logits in prediction for logit in logits]
        query_ids = [
            query_id
            for batch_idx, query_id in enumerate(query_ids)
            for _ in range(len(doc_ids[batch_idx]))
        ]
        doc_ids = [doc_id for doc_ids in doc_ids for doc_id in doc_ids]
        run_df = pd.DataFrame(
            zip(query_ids, doc_ids, scores), columns=["query", "docid", "score"]
        )
        run_df = run_df.sort_values(["query", "score"], ascending=[True, False])
        run_df["rank"] = (
            run_df.groupby("query")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        run_df["q0"] = 0
        run_df["system"] = pl_module.config.name_or_path
        run_df = run_df[RUN_HEADER]
        run_file_path.parent.mkdir(exist_ok=True)
        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"
        run_df.to_csv(run_file_path, header=False, index=False, sep="\t", mode=mode)
