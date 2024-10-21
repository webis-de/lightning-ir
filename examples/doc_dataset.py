from lightning_ir import DocDataset

dataset = DocDataset("msmarco-passage")

print(next(iter(dataset)))
# DocSample(doc_id='0', doc='The presence of communication amid ...')
