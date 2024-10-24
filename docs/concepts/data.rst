.. _concepts-data:

====
Data
====

Lightning IR provides four different datasets for different tasks: The the :py:class:`~lightning_ir.data.dataset.DocDataset` for indexing, :py:class:`~lightning_ir.data.dataset.QueryDataset` for retrieval, the :py:class:`~lightning_ir.data.dataset.TupleDataset` for fine-tuning, and the :py:class:`~lightning_ir.data.dataset.RunDataset` for re-ranking and fine-tuning. The :ref:`datasets` sections provide more in-depth information about how each dataset works and what data it provides. To handle batching and train/validation/test splitting, the datasets should be integrated into a :py:class:`~lightning_ir.data.datamodule.LightningIRDataModule`. See the :ref:`datamodule` section for further details.

By tightly integrating with `ir-datasets <https://ir-datasets.com/>`_, Lightning IR provides easy access to a plethora of popular information retrieval datasets. Simply pass an ``ir-datasets`` id to a dataset class. Custom local datasets are also supported. See :ref:`howto-dataset` for using custom datasets.

.. _datasets:

Datasets
+++++++++

Doc Dataset
-----------

A :py:class:`~lightning_ir.data.dataset.DocDataset` provides access to a set of documents. This is useful for indexing with a :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` where the embeddings of each document are stored in an index that can be used for retrieval. The snippet below demonstrates how to use a :py:class:`~lightning_ir.data.dataset.DocDataset` with an `ir-datasets <https://ir-datasets.com/>`_ dataset.

.. literalinclude:: ../../examples/doc_dataset.py
    :language: python

Query Dataset
-------------

A :py:class:`~lightning_ir.data.dataset.QueryDataset` provides access to a set of queries. This is useful for retrieval with a :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` where the top-k documents are retrieved for each query. The snippet below demonstrates how to use a :py:class:`~lightning_ir.data.dataset.QueryDataset` with an `ir-datasets <https://ir-datasets.com/>`_ dataset.

.. literalinclude:: ../../examples/query_dataset.py
    :language: python

Tuple Dataset
-------------

A :py:class:`~lightning_ir.data.dataset.QueryDataset` provides access to samples consisting of a query and an n-tuple of documents, with each document in a sample also having a corresponding target score. Target scores are relevance assessments and, for example, could have been heuristically sampled, manually assessed, derived from other ranking models for distillation. A :py:class:`~lightning_ir.data.dataset.QueryDataset` dataset is useful for fine-tuning :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` and :py:class:`~lightning_ir.cross_encoder.model.CrossEncoderModel`. The snippet below demonstrates how to use a :py:class:`~lightning_ir.data.dataset.TupleDataset` with an `ir-datasets <https://ir-datasets.com/>`_ dataset.

.. literalinclude:: ../../examples/tuple_dataset.py
    :language: python

Run Dataset
-----------

A :py:class:`~lightning_ir.data.dataset.RunDataset` provides access to a run. A run consists of samples of a query and a list of documents ranked by a relevance score. The dataset may include manual relevance assessments (qrels) which are used to evaluate the effectiveness of retrieval models. This dataset is useful for re-ranking with a :py:class:`~lightning_ir.cross_encoder.model.CrossEncoderModel`. It can also be used for fine-tuning :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` and :py:class:`~lightning_ir.cross_encoder.model.CrossEncoderModel` by sampling tuples from the run. The snippet below demonstrates how to use a :py:class:`~lightning_ir.data.dataset.RunDataset` with an `ir-datasets <https://ir-datasets.com/>`_ dataset.

.. literalinclude:: ../../examples/run_dataset.py
    :language: python

.. _datamodule:

Datamodule
++++++++++

A :py:class:`~lightning_ir.data.datamodule.LightningIRDataModule` conveniently handles the batching and splitting logic necessary to ensure effecient fine-tuning and inference. Depending on the stage (see the :ref:`concepts-trainer` section for details on stages), different combinations of datasets can or must be provided to a datamodule. For fine-tuning, a single ``training_dataset`` in the form of a :py:class:`~lightning_ir.data.dataset.TupleDataset` or :py:class:`~lightning_ir.data.dataset.RunDataset` must be provided and optionally multiple ``inference_datasets`` in the form of :py:class:`~lightning_ir.data.dataset.TupleDataset` or :py:class:`~lightning_ir.data.dataset.RunDataset` can be provided for validation during fine-tuning. For indexing, one or multiple ``inference_datasets`` must be provided in the form of :py:class:`~lightning_ir.data.dataset.DocDataset`. For searching, one or multiple ``inference_datasets`` must be provided in the form of :py:class:`~lightning_ir.data.dataset.QueryDataset`. For re-ranking, one or multiple ``inference_datasets`` must be provided in the form of :py:class:`~lightning_ir.data.dataset.RunDataset`. The snippet below demonstrates how to use a :py:class:`~lightning_ir.data.datamodule.LightningIRDataModule` for fine-tuning with validation using `ir-datasets <https://ir-datasets.com/>`_ datasets.

.. literalinclude:: ../../examples/datamodule.py
    :language: python
