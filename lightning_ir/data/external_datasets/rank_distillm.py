from .ir_datasets_utils import ParquetScoredDocs, register_new_dataset


def register_rank_distillm():
    base_url = "https://huggingface.co/datasets/webis/rank-distillm/resolve/main"

    md5_hashes = {
        "rank-distillm-rankzephyr": (
            "__rankzephyr-colbert-10000-sampled-100__msmarco-passage-train-judged.parquet",
            "38f69a3c8a5ed21c639e882a6c2eff7c",
        ),
        "rank-distillm-set-encoder": (
            "__set-encoder-colbert-all-100__msmarco-passage-train-judged.parquet",
            "a47206da7dc551e3ebd4e5b6866be78a",
        ),
        "rank-distillm-monoelectra": (
            "__monoelectra-colbert-all-100__msmarco-passage-train-judged.parquet",
            "6561f33476a6c8408737f38ea85c848f",
        ),
        "rank-distillm-colbert": (
            "__colbert__msmarco-passage-train-judged.parquet",
            "1e927d52af085516bf5a3de2865809d5",
        ),
    }
    for name, (file_name, expected_md5) in md5_hashes.items():
        dlc_contents = {
            "url": f"{base_url}/{file_name}",
            "expected_md5": expected_md5,
            "cache_path": f"msmarco-passage/train/{name}.parquet",
        }
        register_new_dataset(
            f"msmarco-passage/train/{name}",
            docs="msmarco-passage",
            queries="msmarco-passage/train",
            qrels="msmarco-passage/train",
            scoreddocs=dlc_contents,
            ScoreddocsType=ParquetScoredDocs,
        )
