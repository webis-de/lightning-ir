from lightning_ir import QueryDataset

dataset = QueryDataset("msmarco-passage/trec-dl-2019/judged")

print(next(iter(dataset)))
# QuerySample(query_id='156493', query='do goldfish grow')
