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


def load_constituent(
    constituent: str | None,
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


def register_local(
    dataset_id: str,
    docs: str | None = None,
    queries: str | None = None,
    qrels: str | None = None,
    docpairs: str | None = None,
    scoreddocs: str | None = None,
    qrels_defs: Dict[int, str] | None = None,
):
    if dataset_id in ir_datasets.registry._registered:
        return

    docs = load_constituent(docs, "docs")
    queries = load_constituent(queries, "queries")
    qrels = load_constituent(qrels, "qrels", qrels_defs=qrels_defs if qrels_defs is not None else {})
    docpairs = load_constituent(docpairs, "docpairs")
    scoreddocs = load_constituent(scoreddocs, "scoreddocs")

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


def register_kd_docpairs():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "kd-docpairs"
    cache_path = "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv"
    dlc_contents = {
        "url": (
            "https://zenodo.org/record/4068216/files/bert_cat_ensemble_"
            "msmarcopassage_train_scores_ids.tsv?download=1"
        ),
        "expected_md5": "4d99696386f96a7f1631076bcc53ac3c",
        "cache_path": cache_path,
    }
    file_name = f"{split_id}/{file_id}.tsv"
    register_msmarco(base_id, split_id, file_id, cache_path, dlc_contents, file_name, ScoredDocTuples)


def register_colbert_docpairs():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "colbert-docpairs"
    cache_path = "colbert_64way.json"
    dlc_contents = {
        "url": (
            "https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/" "resolve/main/examples.json?download=true"
        ),
        "expected_md5": "8be0c71e330ac54dcd77fba058d291c7",
        "cache_path": cache_path,
    }
    file_name = f"{split_id}/{file_id}.json"
    register_msmarco(base_id, split_id, file_id, cache_path, dlc_contents, file_name, ScoredDocTuples)


def register_rank_distillm():
    base_id = "msmarco-passage"
    split_id = "train"
    file_id = "rank-distillm/rankzephyr"
    cache_path = "rank-distillm-rankzephyr.run"
    dlc_contents = {
        "url": (
            "https://zenodo.org/records/12528410/files/__rankzephyr-colbert-10000-"
            "sampled-100__msmarco-passage-train-judged.run?download=1"
        ),
        "expected_md5": "49f8dbf2c1ee7a2ca1fe517eda528af6",
        "cache_path": cache_path,
    }
    file_name = f"{split_id}/{file_id}.run"
    register_msmarco(
        base_id,
        split_id,
        file_id,
        cache_path,
        dlc_contents,
        file_name,
        trec.TrecScoredDocs,
    )

    file_id = "rank-distillm/set-encoder"
    cache_path = "rank-distillm-set-encoder.run.gz"
    dlc_contents = {
        "url": (
            "https://zenodo.org/records/12528410/files/__set-encoder-colbert__"
            "msmarco-passage-train-judged.run.gz?download=1"
        ),
        "expected_md5": "1f069d0daa9842a54a858cc660149e1a",
        "cache_path": cache_path,
    }
    file_name = f"{split_id}/{file_id}.run"
    register_msmarco(
        base_id,
        split_id,
        file_id,
        cache_path,
        dlc_contents,
        file_name,
        trec.TrecScoredDocs,
        extract=True,
    )


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


register_kd_docpairs()
register_colbert_docpairs()
register_rank_distillm()
