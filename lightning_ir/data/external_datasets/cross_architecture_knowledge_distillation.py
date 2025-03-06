from .ir_datasets_utils import ScoredDocTuples, register_msmarco


def register_kd_docpairs():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "kd-docpairs"
    cache_path = "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv"
    dlc_contents = {
        "url": (
            "https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1"
        ),
        "expected_md5": "4d99696386f96a7f1631076bcc53ac3c",
        "cache_path": cache_path,
    }
    file_name = f"{file_id}.tsv"
    register_msmarco(base_id, split_id, file_id, cache_path, dlc_contents, file_name, ScoredDocTuples)
