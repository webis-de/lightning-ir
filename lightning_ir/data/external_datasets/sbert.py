import codecs
import json

from lightning_ir.data.external_datasets.ir_datasets_utils import ScoredDocTuples, register_new_dataset


class SBERTScoredDocTuples(ScoredDocTuples):

    def docpairs_iter(self):
        with self._docpairs_dlc.stream() as f:
            f = codecs.getreader("utf8")(f)
            for line in f:
                data = json.loads(line)
                # TODO parse data and yield the docpairs


def register_sbert_docpairs():
    dlc_contents = {
        "url": "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz",
        "expected_md5": "ecf8cafb10197fd7adf4f68aabd15d84",
        "cache_path": "msmarco-passage/train/sbert-docpairs.jsonl.gz",
    }
    register_new_dataset(
        "msmarco-passage/train/sbert-docpairs",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        docpairs=dlc_contents,
        DocpairsType=SBERTScoredDocTuples,
    )


register_sbert_docpairs()

sbert_dataset = SBERTScoredDocTuples("msmarco-passage/train/sbert-docpairs")
print(next(iter(sbert_dataset)))
