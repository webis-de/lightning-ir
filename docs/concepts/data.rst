.. _concepts-data:

====
Data
====

Lightning IR provides four different datasets for different tasks: The the :py:class:`~lightning_ir.data.dataset.DocDataset` for indexing, :py:class:`~lightning_ir.data.dataset.QueryDataset` for retrieval, the :py:class:`~lightning_ir.data.dataset.TupleDataset` for fine-tuning, and the :py:class:`~lightning_ir.data.dataset.RunDataset` for re-ranking and fine-tuning. The ... sections provide more in-depth information about how each dataset works and what data it provides. To handle batching and train/validation/test splitting, the datasets should be integrated into a :py:class:`~lightning_ir.data.datamodule.LightningIRDataModule`. See the ... section for further details.

By tightly integrating with `ir-datasets <https://ir-datasets.com/>`_, Lightning IR provides easy access to a plethora of popular information retrieval datasets. Simply pass an ``ir-datasets`` id to a dataset class ###possesive of class### constructor. Custom local datasets are also supported. See :ref:`howto-dataset` for using custom datasets.

.. _doc-dataset:

Doc Dataset
+++++++++++

A :py:class:`~lightning_ir.data.dataset.DocDataset` provides access to a set of documents. This is useful for indexing with a :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` where the embeddings of each document are stored in an index that can be used for retrieval. The snippet below demonstrates how to use a :py:class:`~lightning_ir.data.dataset.DocDataset` with an `ir-datasets <https://ir-datasets.com/>`_ dataset.

.. literalinclude:: ../../examples/doc_dataset.py
    :language: python

.. _query-dataset:

Query Dataset
+++++++++++++

A :py:class:`~lightning_ir.data.dataset.QueryDataset` provides access to a set of queries. This is useful for retrieval with a :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` where the top-k documents are retrieved for each query. The snippet below demonstrates how to use a :py:class:`~lightning_ir.data.dataset.QueryDataset` with an `ir-datasets <https://ir-datasets.com/>`_ dataset.

.. literalinclude:: ../../examples/query_dataset.py
    :language: python

.. _tuple-dataset:

Tuple Dataset
+++++++++++++

A :py:class:`~lightning_ir.data.dataset.QueryDataset` provides access to a pair of query and n-tuple of documents. Optionally, the dataset can also contain an n-tuple of scores, one score for each document. This dataset is useful for fine-tuning :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` and :py:class:`~lightning_ir.cross_encoder.model.CrossEncoderModel`s. If the dataset contains scores, these can be used for distillation. If no scores are provided, the first document of the n-tuple is usually relevant while the other documents are not relevant to the query. The snippet below demonstrates how to use a :py:class:`~lightning_ir.data.dataset.TupleDataset` with an `ir-datasets <https://ir-datasets.com/>`_ dataset.
