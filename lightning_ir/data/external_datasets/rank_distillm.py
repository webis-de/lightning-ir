from ir_datasets.formats.trec import TrecScoredDocs

from .ir_datasets_utils import register_msmarco


def register_rank_distillm():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "rank-distillm/rankzephyr"
    cache_path = "rank-distillm-rankzephyr.run"
    dlc_contents = {
        "url": (
            "https://zenodo.org/records/12528410/files/__rankzephyr-colbert-10000-"
            "sampled-100__msmarco-passage-train-judged.run?download=1"
        ),
        "expected_md5": "49f8dbf2c1ee7a2ca1fe517eda528af6",
        "cache_path": cache_path,
    }
    file_name = f"{file_id}.run"
    register_msmarco(
        base_id,
        split_id,
        file_id,
        cache_path,
        dlc_contents,
        file_name,
        TrecScoredDocs,
    )

    file_id = "rank-distillm/set-encoder"
    cache_path = "rank-distillm-set-encoder.run.gz"
    dlc_contents = {
        "url": (
            "https://zenodo.org/records/12528410/files/__set-encoder-colbert__"
            "msmarco-passage-train-judged.run.gz?download=1"
        ),
        "expected_md5": "1f069d0daa9842a54a858cc660149e1a",
        "cache_path": cache_path,
    }
    file_name = f"{file_id}.run"
    register_msmarco(
        base_id,
        split_id,
        file_id,
        cache_path,
        dlc_contents,
        file_name,
        TrecScoredDocs,
        extract=True,
    )
