from .ir_datasets_utils import ScoredDocTuples, register_new_dataset


def register_colbert_docpairs():
    dlc_contents = {
        "url": "https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json?download=true",
        "expected_md5": "8be0c71e330ac54dcd77fba058d291c7",
        "cache_path": "msmarco-passage/train/colbert_64way.json",
    }
    register_new_dataset(
        "msmarco-passage/train/colbert-docpairs",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        docpairs=dlc_contents,
        DocpairsType=ScoredDocTuples,
    )
