import pytest
from lightning_ir import DocDataset, QueryDataset

def test_lsr_benchmark_load_docs():
    dataset = DocDataset("lsr-benchmark/msmarco-passage/trec-dl-2019/judged")
    assert dataset is not None
    all_docs = [i for i in dataset]
    assert "8811478" == all_docs[0].doc_id
    assert len(all_docs) == 32123


def test_lsr_benchmark_load_queries():
    dataset = QueryDataset("lsr-benchmark/msmarco-passage/trec-dl-2019/judged")
    assert dataset is not None
    assert dataset.queries is not None
    assert len(dataset.queries) == 43
