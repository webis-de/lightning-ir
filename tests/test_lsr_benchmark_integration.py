from lightning_ir import DocDataset, QueryDataset


def test_lsr_benchmark_load_docs():
    dataset = DocDataset("lsr-benchmark/msmarco-passage/trec-dl-2019/judged")
    assert dataset is not None
    assert "8811478" == next(iter(dataset)).doc_id
    assert len(dataset) == 32123


def test_lsr_benchmark_load_queries():
    dataset = QueryDataset("lsr-benchmark/msmarco-passage/trec-dl-2019/judged")
    assert dataset is not None
    assert dataset.queries is not None
    assert len(dataset.queries) == 43
