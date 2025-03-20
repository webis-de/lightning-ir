.. _ir_datasets: https://ir-datasets.com/

.. _howto-dataset:

====================
Use a Custom Dataset
====================

Lightning IR currently supports all datasets registered with the `ir_datasets`_ library. However, it is also possible to use custom datasets with Lightning IR. `ir_datasets`_ supports five different data types:

- Documents (a collection of documents)
- Queries (a collection of queries)
- Qrels (a collection of relevance judgements for query-document pairs)
- Training n-tuples (a collection of n-tuples consisting of a query and n-1 documents used for training)
- Run Files (a collection of queries and ranked documents)

Depending on your use case, you may need to integrate one or more of these data types. In the following, we will show you locally register datasets with `ir_datasets`_ for easy use in Lightning IR. However, first, we will demonstrate how to integrate custom run files, as these are often generated for datasets already supported by `ir_datasets`_.

Run Files
---------

Integrating your own run files is as simple as providing the run file to the :py:class:`~lightning_ir.data.dataset.RunDataset`. Two types of run files are supported. 

1. The first is a standard TREC run file. When using this format, the file name must conform to a specific naming convention. The file name must correspond to the `ir_datasets`_ dataset id that the run file is associated with. For example, if you have a run file for the TREC Deep Learning 2019, the `ir_datasets`_ dataset id is ``msmarco-passage/trec-dl-2019/judged``. The run file should be named ``msmarco-passage-trec-dl-2019-judged.run``. Optionally, to discern between different run files, you can prefix the file name with meta information surrounded by two underscores, e.g., ``__my-cool-model__msmarco-passage-trec-dl-2019-judged.run``.
2. The second format is a ``.jsonl`` file that not only provides the ``query_id``, ``doc_id``, and the ``score``, but also the actual query and document texts. This format is useful when you want to re-rank a run file but do not want to register the dataset with `ir_datasets`_. The file can optionally contain relevance judgements for easy evaluation. Here is an example of a ``.jsonl`` run file:

.. code-block:: json

    {"query_id": "1", "doc_id": "1", "score": 0.1, "query": "What is the capital of France?", "text": "Paris", "relevance": 1}
    {"query_id": "1", "doc_id": "2", "score": 0.2, "query": "What is the capital of France?", "text": "Berlin", "relevance": 0}
    {"query_id": "2", "doc_id": "1", "score": 0.3, "query": "What is the capital of Germany?", "text": "Berlin", "relevance": 1}
    {"query_id": "2", "doc_id": "2", "score": 0.4, "query": "What is the capital of Germany?", "text": "Paris", "relevance": 0}

Registering a Local Dataset
---------------------------

To integrate a custom dataset it needs to be locally registered with the `ir_datasets`_. Lightning IR provides a :py:class:`~lightning_ir.lightning_utils.callbacks.RegisterLocalDatasetCallback` class to make registering datasets easy. This function takes a dataset id, and optional paths to local files or already valid `ir_datasets`_ dataset ids.

Let's look at an example. Say we wanted to register a new set of training triples for the MS MARCO passage dataset. Our triples file is named ``msmarco-passage-train-triples.tsv`` and has the following format:

.. code-block:: text

    400296  1540783 3518497
    662731  193249  2975302
    238256  4435042 100008

To register this dataset, we can use the following callback. This will copy the documents and queries from the MS MARCO passage dataset and register the new dataset with the id ``msmarco-passage/train/new-train-triples``:

.. code-block:: python

    from lightning_ir import RegisterLocalDatasetCallback

    register_dataset = RegisterLocalDatasetCallback(
        dataset_id='msmarco-passage/train/new-train-triples',
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        docpairs='msmarco-passage-train-triples.tsv'
    )

To use this callback, simply add it to the list of callbacks in your :py:class:`~lightning_ir.main.LightningIRTrainer` and use the new dataset id as the tuples dataset id in the :py:class:`~lightning_ir.data.dataset.TupleDataset`.

.. code-block:: python

    from lightning_ir import TupleDataset, LightningIRTrainer, BiEncoderModule

    train_dataset = TupleDataset(tuples_dataset="msmarco-passage/train/new-train-triples")
    data_module = LightningIRDataModule(train_dataset=train_dataset)
    module = BiEncoderModule() # some bi-encoder model
    trainer = LightningIRTrainer(callbacks=[register_dataset])

    trainer.fit(module, data_module)

If you want to register a dataset that is not based on an existing dataset, you can provide the documents and queries as local files. For example, to register a new dataset with the id ``my-new-dataset``:

.. code-block:: python

    register_dataset = RegisterLocalDatasetCallback(
        dataset_id='my-new-dataset',
        docs="path/to/docs.jsonl",
        queries="path/to/queries.jsonl",
    )