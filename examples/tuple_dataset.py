from lightning_ir import TupleDataset

dataset = TupleDataset("msmarco-passage/train/triples-small")

print(next(iter(dataset)))
# RankSample(
#   query_id='400296',
#   query='is a little caffeine ok during pregnancy',
#   doc_ids=('1540783', '3518497'),
#   docs=(
#       'We don’t know a lot about the effects of caffeine ...',
#       'It is generally safe for pregnant women to eat chocolate ...'
#   ),
#   targets=tensor([1., 0.]),
#   qrels=None
# )
