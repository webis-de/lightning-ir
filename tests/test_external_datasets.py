import pytest

from lightning_ir import RunDataset


@pytest.mark.dataset
@pytest.mark.parametrize(
    "ir_datasets_run_path_or_id",
    [
        "msmarco-passage/train/colbert-docpairs",
        "msmarco-passage/train/kd-docpairs",
        "msmarco-passage/train/rank-distillm-colbert",
        "msmarco-passage/train/rank-distillm-monoelectra",
        "msmarco-passage/train/rank-distillm-rankzephyr",
        "msmarco-passage/train/rank-distillm-set-encoder",
        "msmarco-passage/train/sbert-bm25-docpairs",
        "msmarco-passage/train/sbert-msmarco-distilbert-base-tas-b-docpairs",
        "msmarco-passage/train/sbert-msmarco-distilbert-base-v3-docpairs",
        "msmarco-passage/train/sbert-msmarco-MiniLM-L-6-v3-docpairs",
    ],
)
def test_external_dataset(ir_datasets_run_path_or_id: str):
    dataset = RunDataset(ir_datasets_run_path_or_id).prepare_data()
    assert dataset is not None
