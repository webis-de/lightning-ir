.. _guide-index-types:

=======================
Which Index Type to Use?
=======================

Indexing is only relevant for **bi-encoder** models (cross-encoders score
query–document pairs on the fly). The index type determines the speed–quality
trade-off at search time.

.. code-block:: text

   What kind of bi-encoder embeddings do you have?
   │
   ├── Dense single-vector (DPR)
   │   │
   │   ├── Small collection or need exact results?
   │   │   └── TorchDenseIndexConfig or FaissFlatIndexConfig
   │   │
   │   ├── Large collection, approximate is OK?
   │   │   └── FaissIVFIndexConfig (tune num_centroids)
   │   │
   │   └── Large collection, need compressed index?
   │       └── FaissIVFPQIndexConfig (tune num_centroids, num_subquantizers)
   │
   ├── Dense multi-vector (ColBERT)
   │   │
   │   ├── Small collection or prototyping?
   │   │   └── TorchDenseIndexConfig
   │   │
   │   └── Large collection, production speed?
   │       └── PlaidIndexConfig
   │
   └── Sparse (SPLADE, UniCOIL)
       │
       ├── Simple inverted index?
       │   └── TorchSparseIndexConfig
       │
       └── Fast approximate sparse retrieval?
           └── SeismicIndexConfig

Index Type Comparison
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 16 14 14 14 20

   * - Index Config
     - Search Config
     - Speed
     - Memory
     - Exact?
     - Compatible Models
   * - :py:class:`~lightning_ir.retrieve.pytorch.dense_indexer.TorchDenseIndexConfig`
     - :py:class:`~lightning_ir.retrieve.pytorch.dense_searcher.TorchDenseSearchConfig`
     - Slow
     - High
     - ✅
     - DPR, ColBERT
   * - :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissFlatIndexConfig`
     - :py:class:`~lightning_ir.retrieve.faiss.faiss_searcher.FaissSearchConfig`
     - Medium
     - High
     - ✅
     - DPR, ColBERT
   * - :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissIVFIndexConfig`
     - :py:class:`~lightning_ir.retrieve.faiss.faiss_searcher.FaissSearchConfig`
     - Fast
     - High
     - ❌ (approx.)
     - DPR, ColBERT
   * - :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissIVFPQIndexConfig`
     - :py:class:`~lightning_ir.retrieve.faiss.faiss_searcher.FaissSearchConfig`
     - Fastest
     - Low
     - ❌ (approx.)
     - DPR, ColBERT
   * - :py:class:`~lightning_ir.retrieve.plaid.plaid_indexer.PlaidIndexConfig`
     - :py:class:`~lightning_ir.retrieve.plaid.plaid_searcher.PlaidSearchConfig`
     - Fast
     - Medium
     - ❌ (approx.)
     - ColBERT only
   * - :py:class:`~lightning_ir.retrieve.pytorch.sparse_indexer.TorchSparseIndexConfig`
     - :py:class:`~lightning_ir.retrieve.pytorch.sparse_searcher.TorchSparseSearchConfig`
     - Medium
     - Medium
     - ✅
     - SPLADE, UniCOIL
   * - :py:class:`~lightning_ir.retrieve.seismic.seismic_indexer.SeismicIndexConfig`
     - :py:class:`~lightning_ir.retrieve.seismic.seismic_searcher.SeismicSearchConfig`
     - Fast
     - Medium
     - ❌ (approx.)
     - SPLADE, UniCOIL

.. important::

   The **Search Config** must match the **Index Config** used during indexing.
   You cannot build a FAISS index and search it with a Torch searcher, or
   vice-versa.

Quick Examples
--------------

Index Configs
^^^^^^^^^^^^^

**FAISS IVF index** (approximate nearest-neighbor for large dense collections):

.. code-block:: yaml

   # index-faiss-ivf.yaml
   trainer:
     logger: false
     callbacks:
       - class_path: lightning_ir.IndexCallback
         init_args:
           index_dir: ./my-index
           index_config:
             class_path: lightning_ir.FaissIVFIndexConfig
             init_args:
               num_centroids: 65536
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: webis/bert-bi-encoder
   data:
     class_path: lightning_ir.LightningIRDataModule
     init_args:
       inference_datasets:
         - class_path: lightning_ir.DocDataset
           init_args:
             doc_dataset: msmarco-passage
       inference_batch_size: 256

.. code-block:: python

   from lightning_ir import (
       BiEncoderModule, DocDataset, IndexCallback,
       LightningIRDataModule, LightningIRTrainer,
       FaissIVFIndexConfig,
   )

   module = BiEncoderModule(model_name_or_path="webis/bert-bi-encoder")
   data_module = LightningIRDataModule(
       inference_datasets=[DocDataset("msmarco-passage")],
       inference_batch_size=256,
   )
   callback = IndexCallback(
       index_dir="./my-index",
       index_config=FaissIVFIndexConfig(num_centroids=65536),
   )
   trainer = LightningIRTrainer(
       callbacks=[callback], logger=False, enable_checkpointing=False
   )
   trainer.index(module, data_module)

**Sparse index for SPLADE**:

.. code-block:: yaml

   # index-sparse.yaml
   trainer:
     logger: false
     callbacks:
       - class_path: lightning_ir.IndexCallback
         init_args:
           index_dir: ./splade-index
           index_config:
             class_path: lightning_ir.TorchSparseIndexConfig
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: webis/splade  # hypothetical model
   data:
     class_path: lightning_ir.LightningIRDataModule
     init_args:
       inference_datasets:
         - class_path: lightning_ir.DocDataset
           init_args:
             doc_dataset: msmarco-passage
       inference_batch_size: 256

.. code-block:: python

   from lightning_ir import (
       BiEncoderModule, DocDataset, IndexCallback,
       LightningIRDataModule, LightningIRTrainer,
       TorchSparseIndexConfig,
   )

   module = BiEncoderModule(model_name_or_path="webis/splade")
   data_module = LightningIRDataModule(
       inference_datasets=[DocDataset("msmarco-passage")],
       inference_batch_size=256,
   )
   callback = IndexCallback(
       index_dir="./splade-index",
       index_config=TorchSparseIndexConfig(),
   )
   trainer = LightningIRTrainer(
       callbacks=[callback], logger=False, enable_checkpointing=False
   )
   trainer.index(module, data_module)

Search Configs
^^^^^^^^^^^^^^

**FAISS search** (querying a FAISS IVF index with a dense bi-encoder):

.. code-block:: yaml

   # search-faiss.yaml
   trainer:
     logger: false
     callbacks:
       - class_path: lightning_ir.SearchCallback
         init_args:
           index_dir: ./my-index
           search_config:
             class_path: lightning_ir.FaissSearchConfig
             init_args:
               k: 100
           save_dir: ./runs
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: webis/bert-bi-encoder
   data:
     class_path: lightning_ir.LightningIRDataModule
     init_args:
       inference_datasets:
         - class_path: lightning_ir.QueryDataset
           init_args:
             query_dataset: msmarco-passage/trec-dl-2019/judged
       inference_batch_size: 4

.. code-block:: python

   from lightning_ir import (
       BiEncoderModule, QueryDataset, SearchCallback,
       LightningIRDataModule, LightningIRTrainer,
       FaissSearchConfig,
   )

   module = BiEncoderModule(model_name_or_path="webis/bert-bi-encoder")
   data_module = LightningIRDataModule(
       inference_datasets=[
           QueryDataset("msmarco-passage/trec-dl-2019/judged"),
       ],
       inference_batch_size=4,
   )
   callback = SearchCallback(
       index_dir="./my-index",
       search_config=FaissSearchConfig(k=100),
       save_dir="./runs",
   )
   trainer = LightningIRTrainer(
       callbacks=[callback], logger=False, enable_checkpointing=False
   )
   trainer.search(module, data_module)

**Sparse search for SPLADE**:

.. code-block:: yaml

   # search-sparse.yaml
   trainer:
     logger: false
     callbacks:
       - class_path: lightning_ir.SearchCallback
         init_args:
           index_dir: ./splade-index
           search_config:
             class_path: lightning_ir.TorchSparseSearchConfig
             init_args:
               k: 100
           save_dir: ./runs
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: webis/splade  # hypothetical model
   data:
     class_path: lightning_ir.LightningIRDataModule
     init_args:
       inference_datasets:
         - class_path: lightning_ir.QueryDataset
           init_args:
             query_dataset: msmarco-passage/trec-dl-2019/judged
       inference_batch_size: 4

.. code-block:: python

   from lightning_ir import (
       BiEncoderModule, QueryDataset, SearchCallback,
       LightningIRDataModule, LightningIRTrainer,
       TorchSparseSearchConfig,
   )

   module = BiEncoderModule(model_name_or_path="webis/splade")
   data_module = LightningIRDataModule(
       inference_datasets=[
           QueryDataset("msmarco-passage/trec-dl-2019/judged"),
       ],
       inference_batch_size=4,
   )
   callback = SearchCallback(
       index_dir="./splade-index",
       search_config=TorchSparseSearchConfig(k=100),
       save_dir="./runs",
   )
   trainer = LightningIRTrainer(
       callbacks=[callback], logger=False, enable_checkpointing=False
   )
   trainer.search(module, data_module)

Next steps:

- :doc:`recipes` — See complete end-to-end pipelines
