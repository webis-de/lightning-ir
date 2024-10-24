from lightning_ir import LightningIRDataModule, RunDataset, TupleDataset

train_dataset = TupleDataset("msmarco-passage/train/triples-small")
inference_dataset = RunDataset("msmarco-passage/trec-dl-2019/judged", depth=2)
datamodule = LightningIRDataModule(
    train_dataset=train_dataset,
    train_batch_size=2,
    inference_datasets=[inference_dataset],
    inference_batch_size=2,
)
datamodule.setup("fit")
train_dataloader = datamodule.train_dataloader()
print(next(iter(train_dataloader)))
# TrainBatch(
#   queries=[
#     "is a little caffeine ok during pregnancy",
#     "what fruit is native to australia",
#   ],
#   docs=[
#     (
#       "We don’t know a lot about the effects of caffeine during ...",
#       "It is generally safe for pregnant women to eat chocolate ...",
#     ),
#     (
#       "Passiflora herbertiana. A rare passion fruit native to ...",
#       "The kola nut is the fruit of the kola tree, a genus ...",
#     ),
#   ],
#   query_ids=["400296", "662731"],
#   doc_ids=[("1540783", "3518497"), ("193249", "2975302")],
#   qrels=None,
#   targets=tensor([1.0, 0.0, 1.0, 0.0]),
# )
inference_dataloader = datamodule.inference_dataloader()[0]
print(next(iter(inference_dataloader)))
# RankBatch(
#   queries=["who is robert gray", "cost of interior concrete flooring"],
#   docs=[
#     (
#       "Yellow: combines with blue, lilac, light-cyan, ...",
#       "Salad green: combines with brown, yellowish-brown, ...",
#     ),
#     (
#       "WHAT'S THE DIFFERENCE CONCRETE SLAB VS CONCRETE FLOOR? ...",
#       "If you're trying to figure out what the cost of a concrete ...",
#     ),
#   ],
#   query_ids=["1037798", "104861"],
#   doc_ids=[("7134595", "7134596"), ("841998", "842002")],
#   qrels=[
#     {"query_id": "1037798", "doc_id": "1085628", "iteration": "Q0", "relevance": 0},
#     {"query_id": "1037798", "doc_id": "1308037", "iteration": "Q0", "relevance": 0},
#     ...
#     {"query_id": "104861", "doc_id": "1017088", "iteration": "Q0", "relevance": 0},
#     {"query_id": "104861", "doc_id": "1017092", "iteration": "Q0", "relevance": 2},
#     ...
#   ],
# )
