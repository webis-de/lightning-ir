import codecs
import json
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple, Tuple, Type

import ir_datasets
from ir_datasets.datasets.base import Dataset
from ir_datasets.datasets.nano_beir import parquet_iter
from ir_datasets.formats import (
    BaseDocPairs,
    BaseDocs,
    BaseQrels,
    BaseQueries,
    BaseScoredDocs,
    GenericScoredDoc,
    jsonl,
    trec,
    tsv,
)
from ir_datasets.util import Cache, DownloadConfig

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
    dataset_id: str,
    constituent: Path | str | Dict[str, Any] | None,
    constituent_type: Literal["docs", "queries", "qrels", "scoreddocs", "docpairs"] | Type,
    **kwargs,
) -> Any:
    if constituent is None:
        return None
    if isinstance(constituent, dict):
        constituent_path = Path(constituent["cache_path"])
        cache = _register_and_get_cache(dataset_id, constituent)
    elif constituent in ir_datasets.registry._registered:
        return getattr(ir_datasets.load(constituent), f"{constituent_type}_handler")()
    else:
        constituent_path = Path(constituent)
        cache = Cache(None, constituent_path)
        if not constituent_path.exists():
            raise ValueError(f"unable to load {constituent}, expected an `ir_datasets` id or valid path")
    if isinstance(constituent_type, str):
        suffix = constituent_path.suffixes[0]
        constituent_types = CONSTITUENT_TYPE_MAP[constituent_type]
        if suffix not in constituent_types:
            raise ValueError(f"Unknown file type: {suffix}, expected one of {constituent_types.keys()}")
        ConstituentType = constituent_types[suffix]
    else:
        ConstituentType = constituent_type
    return ConstituentType(cache, **kwargs)


def _register_and_get_cache(dataset_id: str, dlc_contents: Dict[str, Any]) -> Cache:
    extractors = dlc_contents.pop("extractors", [])
    base_id = dataset_id.split("/")[0]
    new_id = dataset_id.removeprefix(base_id + "/")
    base_path = ir_datasets.util.home_path()
    dlc = DownloadConfig.context(base_id, base_path / base_id)
    dlc.contents()[new_id] = dlc_contents
    dataset_dlc = dlc[new_id]
    file_path = Path(dlc_contents["cache_path"])
    for extractor in extractors:
        dataset_dlc = extractor(dataset_dlc)
    return Cache(dataset_dlc, base_path / file_path)


def register_new_dataset(
    dataset_id: str,
    docs: Path | str | Dict[str, str] | None = None,
    DocsType: Type[BaseDocs] | None = None,
    queries: Path | str | Dict[str, str] | None = None,
    QueriesType: Type[BaseQueries] | None = None,
    qrels: Path | str | Dict[str, str] | None = None,
    QrelsType: Type[BaseQrels] | None = None,
    docpairs: Path | str | Dict[str, str] | None = None,
    DocpairsType: Type[BaseDocPairs] | None = None,
    scoreddocs: Path | str | Dict[str, str] | None = None,
    ScoreddocsType: Type[BaseScoredDocs] | None = None,
    qrels_defs: Dict[int, str] | None = None,
):
    if dataset_id in ir_datasets.registry._registered:
        return

    docs = _load_constituent(dataset_id, docs, DocsType or "docs")
    queries = _load_constituent(dataset_id, queries, QueriesType or "queries")
    qrels = _load_constituent(
        dataset_id, qrels, QrelsType or "qrels", qrels_defs=qrels_defs if qrels_defs is not None else {}
    )
    docpairs = _load_constituent(dataset_id, docpairs, DocpairsType or "docpairs")
    scoreddocs = _load_constituent(dataset_id, scoreddocs, ScoreddocsType or "scoreddocs")

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


class ParquetScoredDocs(BaseScoredDocs):
    def __init__(self, scoreddocs_dlc, negate_score=False):
        self._scoreddocs_dlc = scoreddocs_dlc

    def scoreddocs_path(self):
        return self._scoreddocs_dlc.path()

    def scoreddocs_iter(self):
        for d in parquet_iter(self._scoreddocs_dlc.path()):
            yield GenericScoredDoc(d["query-id"], d["corpus-id"], 1)
