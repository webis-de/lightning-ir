import codecs
import json
from functools import partial

from ir_datasets.util import GzipExtract

from lightning_ir.data.external_datasets.ir_datasets_utils import ScoredDocTuple, ScoredDocTuples, register_new_dataset


class SBERTScoredDocTuples(ScoredDocTuples):

    def __init__(self, docpairs_dlc, name):
        super().__init__(docpairs_dlc)
        self.name = name

    def docpairs_iter(self):
        with self._docpairs_dlc.stream() as f:
            f = codecs.getreader("utf8")(f)
            for line in f:
                data = json.loads(line)
                qid = data["qid"]
                pids = []
                scores = []
                for doc in data["pos"]:
                    pids.append(str(doc["pid"]))
                    scores.append(float(doc["ce-score"]))
                for doc in data["neg"][self.name]:
                    pids.append(str(doc["pid"]))
                    scores.append(float(doc["ce-score"]))
                yield ScoredDocTuple(str(qid), tuple(pids), tuple(scores), len(pids))


def register_sbert_docpairs():
    dlc_contents = {
        "url": "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz",
        "expected_md5": "ecf8cafb10197fd7adf4f68aabd15d84",
        "cache_path": "msmarco-passage/train/sbert-docpairs.jsonl",
        "extractors": [GzipExtract],
    }
    for name in [
        "bm25",
        "msmarco-distilbert-base-tas-b",
        "msmarco-distilbert-base-v3",
        "msmarco-MiniLM-L-6-v3",
    ]:
        register_new_dataset(
            f"msmarco-passage/train/sbert-{name}-docpairs",
            docs="msmarco-passage",
            queries="msmarco-passage/train",
            qrels="msmarco-passage/train",
            docpairs=dlc_contents,
            DocpairsType=partial(SBERTScoredDocTuples, name=name),
        )
