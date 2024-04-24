import codecs
import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Sequence, Tuple

import ir_datasets
import torch
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocPairs
from ir_datasets.util import Cache, DownloadConfig
from transformers import BatchEncoding


class ScoredDocTuple(NamedTuple):
    query_id: str
    doc_ids: Tuple[str, ...]
    scores: Tuple[float, ...] | None
    num_docs: int


class ScoredDocTuples(BaseDocPairs):
    def __init__(self, docpairs_dlc):
        self._docpairs_dlc = docpairs_dlc

    def docpairs_path(self):
        return self._docpairs_dlc.path()

    def docpairs_iter(self):
        file_type = None
        if self._docpairs_dlc.path().suffix == ".json":
            file_type = "json"
        elif self._docpairs_dlc.path().suffix in (".tsv", ".run"):
            file_type = "tsv"
        else:
            raise ValueError(f"Unknown file type: {self._docpairs_dlc.path().suffix}")
        with self._docpairs_dlc.stream() as f:
            f = codecs.getreader("utf8")(f)
            for line in f:
                if file_type == "json":
                    data = json.loads(line)
                    qid, *doc_data = data
                    pids, scores = zip(*doc_data)
                    pids = tuple(str(pid) for pid in pids)
                else:
                    cols = line.rstrip().split()
                    pos_score, neg_score, qid, pid1, pid2 = cols
                    pids = (pid1, pid2)
                    scores = (float(pos_score), float(neg_score))
                yield ScoredDocTuple(str(qid), pids, scores, len(pids))

    def docpairs_cls(self):
        return ScoredDocTuple


def register_kd_docpairs():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "kd-docpairs"
    dlc_id = "train/kd-docpairs"
    dlc_contents = {
        "url": (
            "https://zenodo.org/record/4068216/files/bert_cat_ensemble_"
            "msmarcopassage_train_scores_ids.tsv?download=1"
        ),
        "expected_md5": "4d99696386f96a7f1631076bcc53ac3c",
        "cache_path": "train/kd-docpairs",
    }
    file_name = "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv"
    register(base_id, split_id, file_id, dlc_id, dlc_contents, file_name)


def register_colbert_docpairs():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "colbert-docpairs"
    dlc_id = "train/colbert-docpairs"
    dlc_contents = {
        "url": (
            "https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/"
            "resolve/main/examples.json?download=true"
        ),
        "expected_md5": "8be0c71e330ac54dcd77fba058d291c7",
        "cache_path": "train/colbert-docpairs",
    }
    file_name = "colbert_64way.json"
    register(base_id, split_id, file_id, dlc_id, dlc_contents, file_name)


def register(
    base_id: str,
    split_id: str,
    file_id: str,
    dlc_id: str,
    dlc_contents: Dict[str, Any],
    file_name: str,
):
    dataset_id = f"{base_id}/{split_id}/{file_id}"
    if dataset_id in ir_datasets.registry._registered:
        return
    base_path = ir_datasets.util.home_path() / base_id
    dlc = DownloadConfig.context(base_id, base_path)
    dlc._contents[dlc_id] = dlc_contents
    ir_dataset = ir_datasets.load(f"{base_id}/{split_id}")
    collection = ir_dataset.docs_handler()
    queries = ir_dataset.queries_handler()
    qrels = ir_dataset.qrels_handler()
    docpairs = ScoredDocTuples(Cache(dlc[dlc_id], base_path / split_id / file_name))
    dataset = Dataset(collection, queries, qrels, docpairs)
    ir_datasets.registry.register(dataset_id, Dataset(dataset))
    for split in ("train", "val"):
        split_path = Path(
            base_path / split_id / f"__{split}__{dataset_id.replace('/', '-')}.tsv"
        )
        if split_path.exists():
            docpairs = ScoredDocTuples(Cache(None, split_path))
            dataset = Dataset(collection, queries, qrels, docpairs)
            ir_datasets.registry.register(
                f"{base_id}/{split_id}/__{split}__{file_id}", Dataset(dataset)
            )


register_kd_docpairs()
register_colbert_docpairs()


class RunSample(NamedTuple):
    query_id: str
    query: str
    doc_ids: Tuple[str, ...]
    docs: Tuple[str, ...]
    targets: torch.Tensor | None = None
    qrels: Sequence[Dict[str, Any]] | None = None


class QuerySample(NamedTuple):
    query_id: str
    query: str

    @classmethod
    def from_ir_dataset_sample(cls, sample):
        return cls(sample[0], sample[1])


class DocSample(NamedTuple):
    doc_id: str
    doc: str

    @classmethod
    def from_ir_dataset_sample(cls, sample):
        return cls(sample[0], sample.default_text())


class BiEncoderRunBatch(NamedTuple):
    query_ids: Tuple[str, ...]
    query_encoding: BatchEncoding
    doc_ids: Tuple[Tuple[str, ...], ...]
    doc_encoding: BatchEncoding
    targets: torch.Tensor | None = None
    qrels: Dict[str, int] | None = None


class CrossEncoderRunBatch(NamedTuple):
    query_ids: Tuple[str, ...]
    doc_ids: Tuple[Tuple[str, ...], ...]
    encoding: BatchEncoding
    targets: torch.Tensor | None = None
    qrels: Dict[str, int] | None = None


class IndexBatch(NamedTuple):
    doc_ids: Tuple[str, ...]
    doc_encoding: BatchEncoding


class SearchBatch(NamedTuple):
    query_ids: Tuple[str, ...]
    query_encoding: BatchEncoding
