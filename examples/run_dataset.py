from lightning_ir import RunDataset

dataset = RunDataset("msmarco-passage/trec-dl-2019/judged", depth=5)

print(next(iter(dataset)))
# RankSample(
#   query_id='1037798',
#   query='who is robert gray',
#   doc_ids=('7134595', '7134596', ...),
#   docs=(
#       'Yellow: combines with blue, lilac, light-cyan, ...',
#       'Robert Plant Net Worth is $170 Million ... ',
#       ...
#   ),
#   targets=None,
#   qrels=[
#       {'query_id': '1037798', 'doc_id': '1085628', 'iteration': 'Q0', 'relevance': 0},
#        ...
#   ]
# )
