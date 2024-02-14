import codecs
import json
from typing import NamedTuple, Tuple

import ir_datasets
import torch
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocPairs
from ir_datasets.util import Cache, DownloadConfig
from transformers import BatchEncoding


class ScoredDocPair(NamedTuple):
    query_id: str
    doc_id_a: str
    doc_id_b: str
    score_a: float
    score_b: float


class KDDocPairs(BaseDocPairs):
    def __init__(self, docpairs_dlc):
        self._docpairs_dlc = docpairs_dlc

    def docpairs_path(self):
        return self._docpairs_dlc.path()

    def docpairs_iter(self):
        with self._docpairs_dlc.stream() as f:
            f = codecs.getreader("utf8")(f)
            for line in f:
                cols = line.rstrip().split()
                pos_score, neg_score, qid, pid1, pid2 = cols
                pos_score = float(pos_score)
                neg_score = float(neg_score)
                yield ScoredDocPair(qid, pid1, pid2, pos_score, neg_score)

    def docpairs_cls(self):
        return ScoredDocPair


class ScoredDocTuple(NamedTuple):
    query_id: str
    doc_ids: Tuple[str, ...]
    scores: Tuple[float, ...]
    num_docs: int


class ScoredDocTuples(BaseDocPairs):
    def __init__(self, docpairs_dlc):
        self._docpairs_dlc = docpairs_dlc

    def docpairs_path(self):
        return self._docpairs_dlc.path()

    def docpairs_iter(self):
        with self._docpairs_dlc.stream() as f:
            f = codecs.getreader("utf8")(f)
            for line in f:
                data = json.loads(line)
                qid, *doc_data = data
                pids, scores = zip(*doc_data)
                pids = tuple(str(pid) for pid in pids)
                yield ScoredDocTuple(str(qid), tuple(pids), tuple(scores), len(pids))

    def docpairs_cls(self):
        return ScoredDocTuple


def register_kd_docpairs():
    if "msmarco-passage/train/kd-docpairs" in ir_datasets.registry._registered:
        return
    base_path = ir_datasets.util.home_path() / "msmarco-passage"
    dlc = DownloadConfig.context("msmarco-passage", base_path)
    dlc._contents["train/kd-docpairs"] = {
        "url": (
            "https://zenodo.org/record/4068216/files/bert_cat_ensemble_"
            "msmarcopassage_train_scores_ids.tsv?download=1"
        ),
        "expected_md5": "4d99696386f96a7f1631076bcc53ac3c",
        "cache_path": "train/kd-docpairs",
    }
    ir_dataset = ir_datasets.load("msmarco-passage/train")
    collection = ir_dataset.docs_handler()
    queries = ir_dataset.queries_handler()
    qrels = ir_dataset.qrels_handler()
    docpairs = KDDocPairs(
        Cache(dlc["train/kd-docpairs"], base_path / "train" / "kd.run")
    )
    dataset = Dataset(collection, queries, qrels, docpairs)
    ir_datasets.registry.register("msmarco-passage/train/kd-docpairs", Dataset(dataset))


def register_colbert_docpairs():
    if "msmarco-passage/train/colbert-docpairs" in ir_datasets.registry._registered:
        return
    base_path = ir_datasets.util.home_path() / "msmarco-passage"
    dlc = DownloadConfig.context("msmarco-passage", base_path)
    dlc._contents["train/colbert-docpairs"] = {
        "url": (
            "https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/"
            "resolve/main/examples.json?download=true"
        ),
        "expected_md5": "8be0c71e330ac54dcd77fba058d291c7",
        "cache_path": "train/colbert-docpairs",
    }
    ir_dataset = ir_datasets.load("msmarco-passage/train")
    collection = ir_dataset.docs_handler()
    queries = ir_dataset.queries_handler()
    qrels = ir_dataset.qrels_handler()
    docpairs = ScoredDocTuples(
        Cache(dlc["train/colbert-docpairs"], base_path / "train" / "colbert_64way.json")
    )
    dataset = Dataset(collection, queries, qrels, docpairs)
    ir_datasets.registry.register(
        "msmarco-passage/train/colbert-docpairs", Dataset(dataset)
    )


register_kd_docpairs()
register_colbert_docpairs()


class Sample(NamedTuple):
    query_id: str
    query: str
    doc_ids: Tuple[str, ...]
    docs: Tuple[str, ...]
    targets: Tuple[float, ...]
    relevance: Tuple[float, ...] | None = None


class Batch(NamedTuple):
    query_ids: Tuple[str, ...]
    query_encoding: BatchEncoding
    doc_ids: Tuple[Tuple[str, ...], ...]
    doc_encoding: BatchEncoding
    targets: torch.Tensor
    relevance: torch.Tensor | None = None
