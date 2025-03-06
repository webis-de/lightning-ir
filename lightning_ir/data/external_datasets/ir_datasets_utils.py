import codecs
import json
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple, Tuple, Type

import ir_datasets
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocPairs, jsonl, trec, tsv
from ir_datasets.util import Cache, DownloadConfig, GzipExtract

CONSTITUENT_TYPE_MAP: Dict[str, Dict[str, Type]] = {
    "docs": {
        ".json": jsonl.JsonlDocs,
        ".jsonl": jsonl.JsonlDocs,
        ".tsv": tsv.TsvDocs,
    },
    "queries": {
        ".json": jsonl.JsonlQueries,
        ".jsonl": jsonl.JsonlQueries,
        ".tsv": tsv.TsvQueries,
    },
    "qrels": {".tsv": trec.TrecQrels, ".qrels": trec.TrecQrels},
    "scoreddocs": {".run": trec.TrecScoredDocs, ".tsv": trec.TrecScoredDocs},
    "docpairs": {".tsv": tsv.TsvDocPairs},
}


def _load_constituent(
    constituent: Path | str | None,
    constituent_type: Literal["docs", "queries", "qrels", "scoreddocs", "docpairs"],
    **kwargs,
) -> Any:
    if constituent is None:
        return None
    if constituent in ir_datasets.registry._registered:
        return getattr(ir_datasets.load(constituent), f"{constituent_type}_handler")
    constituent_path = Path(constituent)
    if not constituent_path.exists():
        raise ValueError(f"unable to load {constituent}, expected an `ir_datasets` id or valid path")
    suffix = constituent_path.suffixes[0]
    constituent_types = CONSTITUENT_TYPE_MAP[constituent_type]
    if suffix not in constituent_types:
        raise ValueError(f"Unknown file type: {suffix}, expected one of {constituent_types.keys()}")
    ConstituentType = constituent_types[suffix]
    return ConstituentType(Cache(None, constituent_path), **kwargs)


def register_local_dataset(
    dataset_id: str,
    docs: Path | str | None = None,
    queries: Path | str | None = None,
    qrels: Path | str | None = None,
    docpairs: Path | str | None = None,
    scoreddocs: Path | str | None = None,
    qrels_defs: Dict[int, str] | None = None,
):
    if dataset_id in ir_datasets.registry._registered:
        return

    docs = _load_constituent(docs, "docs")
    queries = _load_constituent(queries, "queries")
    qrels = _load_constituent(qrels, "qrels", qrels_defs=qrels_defs if qrels_defs is not None else {})
    docpairs = _load_constituent(docpairs, "docpairs")
    scoreddocs = _load_constituent(scoreddocs, "scoreddocs")

    ir_datasets.registry.register(dataset_id, Dataset(docs, queries, qrels, docpairs, scoreddocs))


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


def register_msmarco(
    base_id: str,
    split_id: str,
    file_id: str,
    cache_path: str,
    dlc_contents: Dict[str, Any],
    file_name: str,
    ConstituentType: Type,
    extract: bool = False,
):
    dataset_id = f"{base_id}/{split_id}/{file_id}"
    if dataset_id in ir_datasets.registry._registered:
        return
    base_path = ir_datasets.util.home_path() / base_id
    dlc = DownloadConfig.context(base_id, base_path)
    dlc._contents[cache_path] = dlc_contents
    ir_dataset = ir_datasets.load(f"{base_id}/{split_id}")
    collection = ir_dataset.docs_handler()
    queries = ir_dataset.queries_handler()
    qrels = ir_dataset.qrels_handler()
    _dlc = dlc[cache_path]
    if extract:
        _dlc = GzipExtract(_dlc)
    constituent = ConstituentType(Cache(_dlc, base_path / split_id / file_name))
    dataset = Dataset(collection, queries, qrels, constituent)
    ir_datasets.registry.register(dataset_id, Dataset(dataset))
