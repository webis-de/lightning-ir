from .ir_datasets_utils import ScoredDocTuples, register_msmarco


def register_colbert_docpairs():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "colbert-docpairs"
    cache_path = "colbert_64way.json"
    dlc_contents = {
        "url": "https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json?download=true",
        "expected_md5": "8be0c71e330ac54dcd77fba058d291c7",
        "cache_path": cache_path,
    }
    file_name = f"{file_id}.json"
    register_msmarco(base_id, split_id, file_id, cache_path, dlc_contents, file_name, ScoredDocTuples)
