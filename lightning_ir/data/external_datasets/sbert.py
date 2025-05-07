import codecs

from .ir_datasets_utils import ScoredDocTuples, register_new_dataset

# import json


class SBERTScordDocTuples(ScoredDocTuples):

    def docpairs_iter(self):
        with self._docpairs_dlc.stream() as f:
            f = codecs.getreader("utf8")(f)
            for line in f:
                pass
                # data = json.loads(line)
                # TODO parse data


def _register_sbert_docpairs():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "sbert-docpairs"
    cache_path = "sbert.jsonl.gz"
    dlc_contents = {
        "url": "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz",
        "expected_md5": "ecf8cafb10197fd7adf4f68aabd15d84",
        "cache_path": cache_path,
    }
    file_name = f"{file_id}.jsonl"
    register_new_dataset(
        base_id, split_id, file_id, cache_path, dlc_contents, file_name, SBERTScordDocTuples, extract=True
    )
