from .ir_datasets_utils import ParquetScoredDocs, register_new_dataset


def register_rank_distillm():

    base_url = "https://huggingface.co/datasets/webis/rank-distillm/resolve/main/"

    dlc_contents = {
        "url": f"{base_url}__rankzephyr-colbert-10000-sampled-100__msmarco-passage-train-judged.parquet",
        "expected_md5": "02a245c712b4ea1804d0cb504005c0e2",
        "cache_path": "msmarco-passage/train/rank-distillm-rankzephyr.parquet",
    }
    register_new_dataset(
        "msmarco-passage/train/rank-distillm-rankzephyr",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        scoreddocs=dlc_contents,
        ScoreddocsType=ParquetScoredDocs,
    )

    dlc_contents = {
        "url": f"{base_url}__set-encoder-colbert-all-100__msmarco-passage-train-judged.parquet",
        "expected_md5": "a47206da7dc551e3ebd4e5b6866be78a",
        "cache_path": "msmarco-passage/train/rank-distillm-set-encoder.parquet",
    }
    register_new_dataset(
        "msmarco-passage/train/rank-distillm-set-encoder",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        scoreddocs=dlc_contents,
        ScoreddocsType=ParquetScoredDocs,
    )

    dlc_contents = {
        "url": f"{base_url}__monoelectra-colbert-all-100__msmarco-passage-train-judged.parquet",
        "expected_md5": "6561f33476a6c8408737f38ea85c848f",
        "cache_path": "msmarco-passage/train/rank-distillm-monoelectra.parquet",
    }
    register_new_dataset(
        "msmarco-passage/train/rank-distillm-monoelectra",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        scoreddocs=dlc_contents,
        ScoreddocsType=ParquetScoredDocs,
    )

    dlc_contents = {
        "url": f"{base_url}__colbert__msmarco-passage-train-judged.parquet",
        "expected_md5": "1e927d52af085516bf5a3de2865809d5",
        "cache_path": "msmarco-passage/train/rank-distillm-colbert.parquet",
    }
    register_new_dataset(
        "msmarco-passage/train/rank-distillm-colbert",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        scoreddocs=dlc_contents,
        ScoreddocsType=ParquetScoredDocs,
    )
