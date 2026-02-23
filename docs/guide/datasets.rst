.. _guide-datasets:

===========================
Which Dataset Format to Use?
===========================

Lightning IR provides four dataset classes. The right one depends on your
workflow and the shape of your data.

.. code-block:: text

   What are you trying to do?
   │
   ├── Fine-tune a model (fit)
   │   │
   │   ├── Have query + positive + negative triples?
   │   │   └── TupleDataset
   │   │       (uses an ir_datasets ID, e.g. "msmarco-passage/train/triples-small")
   │   │
   │   └── Have a run file with ranked docs and teacher scores?
   │       └── RunDataset (targets: score, sampling_strategy: random)
   │
   ├── Index documents (index)
   │   └── DocDataset
   │       (uses an ir_datasets ID, e.g. "msmarco-passage")
   │
   ├── Search / retrieve (search)
   │   └── QueryDataset
   │       (uses an ir_datasets ID, e.g. "msmarco-passage/trec-dl-2019/judged")
   │
   └── Re-rank (re_rank)
       └── RunDataset
           (path to a TREC-format run file or an ir_datasets ID)

Dataset Class Reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Dataset
     - Workflow
     - Description
   * - :py:class:`~lightning_ir.data.dataset.TupleDataset`
     - ``fit``
     - Iterates over (query, doc₁, doc₂, …) tuples with relevance targets.
       Backed by `ir_datasets <https://ir-datasets.com/>`_.
   * - :py:class:`~lightning_ir.data.dataset.RunDataset`
     - ``fit``, ``re_rank``
     - Loads a ranked list of documents per query from a TREC-format run file
       or an ir_datasets ID. Key parameters: ``depth`` (max rank to load),
       ``sample_size`` (docs per query), ``sampling_strategy``
       (``top`` or ``random``), ``targets`` (``relevance`` or ``score``).
   * - :py:class:`~lightning_ir.data.dataset.DocDataset`
     - ``index``
     - Iterates over all documents in a collection. Backed by ir_datasets.
   * - :py:class:`~lightning_ir.data.dataset.QueryDataset`
     - ``search``
     - Iterates over queries in a dataset split. Backed by ir_datasets.

.. tip::

   When using :py:class:`~lightning_ir.data.dataset.RunDataset` for **training** (knowledge distillation), set
   ``sampling_strategy: random`` so the model sees diverse negatives, and
   ``targets: score`` to use the teacher's relevance scores.

   When using :py:class:`~lightning_ir.data.dataset.RunDataset` for **re-ranking** (inference), set
   ``sampling_strategy: top`` and increase ``depth`` / ``sample_size``
   to cover the full candidate list.

Next steps:

- :doc:`loss-functions` — Choose a loss function for training
- :doc:`recipes` — See complete end-to-end pipelines
