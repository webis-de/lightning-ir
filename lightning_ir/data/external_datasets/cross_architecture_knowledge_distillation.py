from .ir_datasets_utils import ScoredDocTuples, register_new_dataset


def register_kd_docpairs():
    dlc_contents = {
        "url": (
            "https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1"
        ),
        "expected_md5": "4d99696386f96a7f1631076bcc53ac3c",
        "cache_path": "msmarco-passage/train/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv",
    }
    register_new_dataset(
        "msmarco-passage/train/kd-docpairs",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        docpairs=dlc_contents,
        DocpairsType=ScoredDocTuples,
    )
