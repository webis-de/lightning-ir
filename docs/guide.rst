.. _guide:

==============
Decision Guide
==============

This guide helps you navigate Lightning IR's configuration space. It is structured
as a series of decision trees: start with *what you want to do*, then follow the
branches to pick the right model architecture, index type, loss function, and data
format. Each section ends with concrete CLI and Python examples you can copy and
adapt.

-----------------------
What Do You Want to Do?
-----------------------

Start here. Lightning IR supports four top-level workflows, exposed as
sub-commands of the ``lightning-ir`` CLI and as methods on
:py:class:`~lightning_ir.main.LightningIRTrainer`.

.. code-block:: text

                       ┌──────────────────────┐
                       │  What is your goal?  │
                       └──────────┬───────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │    Fine-Tune     │  │   Retrieve docs  │  │ Improve existing │
   │     a model      │  │   from a large   │  │     rankings     │
   │                  │  │    collection    │  │                  │
   │      ► fit       │  │     ► index      │  │    ► re_rank     │
   │                  │  │     ► search     │  │                  │
   └──────────────────┘  └──────────────────┘  └──────────────────┘

The table below summarizes the key ingredients for each workflow.

.. list-table::
   :header-rows: 1
   :widths: 12 18 20 25 25

   * - Workflow
     - CLI Sub-command
     - Module Type
     - Dataset Type
     - Required Callback
   * - Fine-tune a model
     - ``fit``
     - :py:class:`~lightning_ir.bi_encoder.bi_encoder_module.BiEncoderModule` **or** :py:class:`~lightning_ir.cross_encoder.cross_encoder_module.CrossEncoderModule`
     - :py:class:`~lightning_ir.data.dataset.TupleDataset` or :py:class:`~lightning_ir.data.dataset.RunDataset` (train)
     - *(none — optional ModelCheckpoint)*
   * - Index documents
     - ``index``
     - :py:class:`~lightning_ir.bi_encoder.bi_encoder_module.BiEncoderModule`
     - :py:class:`~lightning_ir.data.dataset.DocDataset`
     - :py:class:`~lightning_ir.callbacks.callbacks.IndexCallback`
   * - Search (retrieve)
     - ``search``
     - :py:class:`~lightning_ir.bi_encoder.bi_encoder_module.BiEncoderModule`
     - :py:class:`~lightning_ir.data.dataset.QueryDataset`
     - :py:class:`~lightning_ir.callbacks.callbacks.SearchCallback`
   * - Re-rank
     - ``re_rank``
     - :py:class:`~lightning_ir.bi_encoder.bi_encoder_module.BiEncoderModule` **or** :py:class:`~lightning_ir.cross_encoder.cross_encoder_module.CrossEncoderModule`
     - :py:class:`~lightning_ir.data.dataset.RunDataset`
     - :py:class:`~lightning_ir.callbacks.callbacks.ReRankCallback`

.. tip::

   A typical end-to-end pipeline chains several workflows:

   1. **fit** — Fine-tune a model
   2. **index** — Encode all documents into an index (bi-encoder only)
   3. **search** — Retrieve candidate documents for queries
   4. **re_rank** — Re-score candidates with a more powerful model (often a cross-encoder)

   You can enter the pipeline at any point. For example, if you already have a
   fine-tuned model from the :ref:`model-zoo`, skip straight to **index** or
   **re_rank**.

---------------------------------
Which Model Architecture to Use?
---------------------------------

This is usually the most impactful decision. The diagram below captures the
main trade-offs.

.. code-block:: text

   Do you need to retrieve from a large collection (millions of docs)?
   │
   ├── YES ─► Use a Bi-Encoder
   │          │
   │          ├── Need sparse / lexical matching with term expansion?
   │          │   └── YES ─► SPLADE (SpladeConfig)
   │          │
   │          ├── Need highest bi-encoder quality via token-level matching?
   │          │   └── YES ─► ColBERT (ColConfig)
   │          │
   │          └── Want simplest dense single-vector retrieval?
   │              └── YES ─► DPR (DprConfig)
   │
   └── NO ─► You only need to re-rank an existing candidate list
             │
             ├── Pointwise scoring (one doc at a time)?
             │   └── YES ─► MonoEncoder (MonoConfig)
             │
             └── Listwise scoring (all candidates at once)?
                 └── YES ─► SetEncoder (SetEncoderConfig)

Architecture Comparison
+++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 14 14 12 12 14 14 20

   * - Architecture
     - Config Class
     - Encoding
     - Vectors
     - Retrieval
     - Re-ranking
     - Key Trade-off
   * - **DPR**
     - :py:class:`~lightning_ir.models.bi_encoders.dpr.DprConfig`
     - Separate
     - Single dense
     - ✅
     - ✅
     - Fastest index & search; lower quality
   * - **SPLADE**
     - :py:class:`~lightning_ir.models.bi_encoders.splade.SpladeConfig`
     - Separate
     - Single sparse
     - ✅
     - ✅
     - Interpretable lexical matching; needs regularization
   * - **ColBERT**
     - :py:class:`~lightning_ir.models.bi_encoders.col.ColConfig`
     - Separate
     - Multi dense
     - ✅
     - ✅
     - Best bi-encoder quality; larger index
   * - **MonoEncoder**
     - :py:class:`~lightning_ir.models.cross_encoders.mono.MonoConfig`
     - Joint
     - —
     - ❌
     - ✅
     - Highest quality; cannot index
   * - **SetEncoder**
     - :py:class:`~lightning_ir.models.cross_encoders.set_encoder.SetEncoderConfig`
     - Joint (listwise)
     - —
     - ❌
     - ✅
     - Sees all candidates at once; highest re-rank quality

.. note::

   Lightning IR also supports several other bi-encoder variants (:py:class:`~lightning_ir.models.bi_encoders.coil.CoilConfig`,
   :py:class:`~lightning_ir.models.bi_encoders.mvr.MvrConfig`) and external models such as XTR for advanced use cases. See
   the :ref:`concepts-model` page and the :py:mod:`~lightning_ir.models` API
   reference for a full list.

Quick Examples: Picking a Model
+++++++++++++++++++++++++++++++

**DPR bi-encoder** — simplest dense retrieval:

.. code-block:: yaml

   # model-dpr.yaml
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: bert-base-uncased
       config:
         class_path: lightning_ir.models.DprConfig

.. code-block:: python

   from lightning_ir import BiEncoderModule
   from lightning_ir.models import DprConfig

   module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=DprConfig(),
   )

**ColBERT** — multi-vector late interaction:

.. code-block:: yaml

   # model-colbert.yaml
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: bert-base-uncased
       config:
         class_path: lightning_ir.models.ColConfig
         init_args:
           similarity_function: dot
           query_aggregation_function: sum
           query_expansion: true
           query_length: 32
           doc_length: 256
           normalization_strategy: l2
           embedding_dim: 128
           projection: linear_no_bias
           add_marker_tokens: true

.. code-block:: python

   from lightning_ir import BiEncoderModule
   from lightning_ir.models import ColConfig

   module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=ColConfig(
           similarity_function="dot",
           query_aggregation_function="sum",
           query_expansion=True,
           query_length=32,
           doc_length=256,
           normalization_strategy="l2",
           embedding_dim=128,
           projection="linear_no_bias",
           add_marker_tokens=True,
       ),
   )

**SPLADE** — learned sparse retrieval:

.. code-block:: yaml

   # model-splade.yaml
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: bert-base-uncased
       config:
         class_path: lightning_ir.models.SpladeConfig

.. code-block:: python

   from lightning_ir import BiEncoderModule
   from lightning_ir.models import SpladeConfig

   module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=SpladeConfig(),
   )

**Cross-encoder (MonoEncoder)** — highest quality re-ranking:

.. code-block:: yaml

   # model-cross-encoder.yaml
   model:
     class_path: lightning_ir.CrossEncoderModule
     init_args:
       model_name_or_path: webis/monoelectra-base

.. code-block:: python

   from lightning_ir import CrossEncoderModule

   module = CrossEncoderModule(
       model_name_or_path="webis/monoelectra-base",
   )


-----------------------
Which Index Type to Use?
-----------------------

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
+++++++++++++++++++++

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

Quick Examples: Indexing & Searching
++++++++++++++++++++++++++++++++++++

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


-------------------------------
Which Loss Function to Use?
-------------------------------

The choice of loss function depends on your training data format and your
training objective.

.. code-block:: text

   What does your training data look like?
   │
   ├── Triples (query, positive doc, negative doc) — TupleDataset
   │   │
   │   ├── Want a simple pairwise objective?
   │   │   └── RankNet or ConstantMarginMSE
   │   │
   │   ├── Want contrastive learning with in-batch negatives?
   │   │   └── InBatchCrossEntropy
   │   │
   │   └── Want to directly optimize a ranking metric (e.g., nDCG)?
   │       └── ApproxNDCG or ApproxMRR
   │
   ├── Ranked list with teacher scores — RunDataset (targets: score)
   │   │
   │   ├── Knowledge distillation from a teacher model?
   │   │   └── KLDivergence or RankNet
   │   │
   │   └── Want to match teacher ranking distribution?
   │       └── InfoNCE
   │
   └── Training a sparse model (SPLADE)?
       └── Add a regularization loss alongside your main loss:
           FLOPSRegularization (+ GenericConstantSchedulerWithLinearWarmup callback)

Loss Function Reference
+++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Loss
     - Category
     - When to Use
   * - :py:class:`~lightning_ir.loss.pairwise.RankNet`
     - Pairwise
     - Default choice for training with triples (pos/neg pairs). Optimizes
       pairwise ranking accuracy.
   * - :py:class:`~lightning_ir.loss.pairwise.ConstantMarginMSE`
     - Pairwise
     - Pairwise MSE with a fixed margin between positive and negative scores.
   * - :py:class:`~lightning_ir.loss.pairwise.SupervisedMarginMSE`
     - Pairwise
     - Pairwise MSE where the margin is derived from relevance labels.
   * - :py:class:`~lightning_ir.loss.listwise.KLDivergence`
     - Listwise
     - Knowledge distillation from a teacher model's score distribution.
       Requires :py:class:`~lightning_ir.data.dataset.RunDataset` with ``targets: score``.
   * - :py:class:`~lightning_ir.loss.listwise.InfoNCE`
     - Listwise
     - Contrastive loss over a list of scored candidates.
   * - :py:class:`~lightning_ir.loss.listwise.PearsonCorrelation`
     - Listwise
     - Optimizes correlation between predicted and target scores.
   * - :py:class:`~lightning_ir.loss.approximate.ApproxNDCG`
     - Approximate
     - Differentiable approximation of nDCG. Directly optimizes the target
       metric.
   * - :py:class:`~lightning_ir.loss.approximate.ApproxMRR`
     - Approximate
     - Differentiable approximation of MRR.
   * - :py:class:`~lightning_ir.loss.approximate.ApproxRankMSE`
     - Approximate
     - MSE on approximate rank positions.
   * - :py:class:`~lightning_ir.loss.in_batch.InBatchCrossEntropy`
     - In-batch
     - Uses other queries' positives as negatives within a batch. Very
       effective with large batch sizes.
   * - :py:class:`~lightning_ir.loss.in_batch.ScoreBasedInBatchCrossEntropy`
     - In-batch
     - In-batch negatives weighted by teacher scores.
   * - :py:class:`~lightning_ir.loss.regularization.FLOPSRegularization`
     - Regularization
     - Encourages sparsity in SPLADE embeddings. **Always** combine with a
       primary loss and a warmup scheduler.
   * - :py:class:`~lightning_ir.loss.regularization.L1Regularization`
     - Regularization
     - L1 penalty on embedding values.
   * - :py:class:`~lightning_ir.loss.regularization.L2Regularization`
     - Regularization
     - L2 penalty on embedding values.

Quick Example: Combining Losses for SPLADE
++++++++++++++++++++++++++++++++++++++++++

SPLADE models typically require a primary ranking loss **plus** a FLOPS
regularization loss with a warmup schedule:

.. code-block:: yaml

   # splade-training.yaml
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: bert-base-uncased
       config:
         class_path: lightning_ir.models.SpladeConfig
       loss_functions:
         - lightning_ir.InBatchCrossEntropy
         - class_path: lightning_ir.FLOPSRegularization
           init_args:
             query_weight: 0.0008
             doc_weight: 0.0006
   trainer:
     max_steps: 100_000
     callbacks:
       - class_path: lightning_ir.GenericConstantSchedulerWithLinearWarmup
         init_args:
           keys:
             - loss_functions.1.query_weight
             - loss_functions.1.doc_weight
           num_warmup_steps: 20_000
           num_delay_steps: 50_000

.. code-block:: python

   from lightning_ir import (
       BiEncoderModule, LightningIRTrainer, LightningIRDataModule,
       TupleDataset, InBatchCrossEntropy, FLOPSRegularization,
       GenericConstantSchedulerWithLinearWarmup,
   )
   from lightning_ir.models import SpladeConfig
   from torch.optim import AdamW

   module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=SpladeConfig(),
       loss_functions=[
           InBatchCrossEntropy(),
           FLOPSRegularization(query_weight=0.0008, doc_weight=0.0006),
       ],
   )
   module.set_optimizer(AdamW, lr=1e-5)

   data_module = LightningIRDataModule(
       train_dataset=TupleDataset("msmarco-passage/train/triples-small"),
       train_batch_size=32,
   )
   scheduler = GenericConstantSchedulerWithLinearWarmup(
       keys=["loss_functions.1.query_weight", "loss_functions.1.doc_weight"],
       num_warmup_steps=20_000,
       num_delay_steps=50_000,
   )
   trainer = LightningIRTrainer(max_steps=100_000, callbacks=[scheduler])
   trainer.fit(module, data_module)

Quick Example: Knowledge Distillation
++++++++++++++++++++++++++++++++++++++

To distill from a teacher model's run file scores into a student model:

.. code-block:: yaml

   # distillation.yaml
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: bert-base-uncased
       config:
         class_path: lightning_ir.models.DprConfig
       loss_functions:
         - lightning_ir.KLDivergence
   data:
     class_path: lightning_ir.LightningIRDataModule
     init_args:
       train_dataset:
         class_path: lightning_ir.RunDataset
         init_args:
           run_path_or_id: msmarco-passage/train/rank-distillm/set-encoder
           depth: 50
           sample_size: 8
           sampling_strategy: random
           targets: score
       train_batch_size: 16

.. code-block:: python

   from lightning_ir import (
       BiEncoderModule, LightningIRTrainer, LightningIRDataModule,
       RunDataset, KLDivergence,
   )
   from lightning_ir.models import DprConfig
   from torch.optim import AdamW

   module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=DprConfig(),
       loss_functions=[KLDivergence()],
   )
   module.set_optimizer(AdamW, lr=1e-5)

   data_module = LightningIRDataModule(
       train_dataset=RunDataset(
           run_path_or_id="msmarco-passage/train/rank-distillm/set-encoder",
           depth=50,
           sample_size=8,
           sampling_strategy="random",
           targets="score",
       ),
       train_batch_size=16,
   )
   trainer = LightningIRTrainer(max_steps=50_000)
   trainer.fit(module, data_module)


-------------------------------
Which Dataset Format to Use?
-------------------------------

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
+++++++++++++++++++++++

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


-----------------------------------
End-to-End Recipes
-----------------------------------

The following recipes chain together the decisions above into complete,
copy-pasteable pipelines. Each recipe shows both the CLI (YAML) and
programmatic (Python) approach.

Recipe 1: DPR Dense Retrieval Pipeline
++++++++++++++++++++++++++++++++++++++

*Goal:* Fine-tune a simple dense bi-encoder, index a collection, search, then
re-rank with a cross-encoder.

**Step 1 — Fine-tune the DPR model:**

.. code-block:: bash

   lightning-ir fit --config recipe-dpr-fit.yaml

.. collapse:: recipe-dpr-fit.yaml

   .. code-block:: yaml

      trainer:
        max_steps: 100_000
        precision: bf16-mixed
        accumulate_grad_batches: 4
        gradient_clip_val: 1
      model:
        class_path: lightning_ir.BiEncoderModule
        init_args:
          model_name_or_path: bert-base-uncased
          config:
            class_path: lightning_ir.models.DprConfig
          loss_functions:
            - lightning_ir.RankNet
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          train_dataset:
            class_path: lightning_ir.TupleDataset
            init_args:
              tuples_dataset: msmarco-passage/train/triples-small
          train_batch_size: 32
      optimizer:
        class_path: torch.optim.AdamW
        init_args:
          lr: 1e-5

.. collapse:: recipe_dpr_fit.py

   .. code-block:: python

      from torch.optim import AdamW
      from lightning_ir import (
          BiEncoderModule, LightningIRDataModule,
          LightningIRTrainer, RankNet, TupleDataset,
      )
      from lightning_ir.models import DprConfig

      module = BiEncoderModule(
          model_name_or_path="bert-base-uncased",
          config=DprConfig(),
          loss_functions=[RankNet()],
      )
      module.set_optimizer(AdamW, lr=1e-5)

      data_module = LightningIRDataModule(
          train_dataset=TupleDataset("msmarco-passage/train/triples-small"),
          train_batch_size=32,
      )
      trainer = LightningIRTrainer(
          max_steps=100_000,
          precision="bf16-mixed",
          accumulate_grad_batches=4,
          gradient_clip_val=1,
      )
      trainer.fit(module, data_module)

**Step 2 — Index the collection:**

.. code-block:: bash

   lightning-ir index --config recipe-dpr-index.yaml

.. collapse:: recipe-dpr-index.yaml

   .. code-block:: yaml

      trainer:
        logger: false
        callbacks:
          - class_path: lightning_ir.IndexCallback
            init_args:
              index_dir: ./msmarco-passage-index
              index_config:
                class_path: lightning_ir.FaissFlatIndexConfig
      model:
        class_path: lightning_ir.BiEncoderModule
        init_args:
          model_name_or_path: ./my-dpr-checkpoint  # or a Model Zoo model
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          inference_datasets:
            - class_path: lightning_ir.DocDataset
              init_args:
                doc_dataset: msmarco-passage
          inference_batch_size: 256

.. collapse:: recipe_dpr_index.py

   .. code-block:: python

      from lightning_ir import (
          BiEncoderModule, DocDataset, IndexCallback,
          LightningIRDataModule, LightningIRTrainer,
          FaissFlatIndexConfig,
      )

      module = BiEncoderModule(model_name_or_path="./my-dpr-checkpoint")
      data_module = LightningIRDataModule(
          inference_datasets=[DocDataset("msmarco-passage")],
          inference_batch_size=256,
      )
      callback = IndexCallback(
          index_dir="./msmarco-passage-index",
          index_config=FaissFlatIndexConfig(),
      )
      trainer = LightningIRTrainer(
          callbacks=[callback], logger=False, enable_checkpointing=False
      )
      trainer.index(module, data_module)

**Step 3 — Search:**

.. code-block:: bash

   lightning-ir search --config recipe-dpr-search.yaml

.. collapse:: recipe-dpr-search.yaml

   .. code-block:: yaml

      trainer:
        logger: false
        callbacks:
          - class_path: lightning_ir.SearchCallback
            init_args:
              index_dir: ./msmarco-passage-index
              search_config:
                class_path: lightning_ir.FaissSearchConfig
                init_args:
                  k: 100
              save_dir: ./runs
      model:
        class_path: lightning_ir.BiEncoderModule
        init_args:
          model_name_or_path: ./my-dpr-checkpoint
          evaluation_metrics:
            - nDCG@10
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          inference_datasets:
            - class_path: lightning_ir.QueryDataset
              init_args:
                query_dataset: msmarco-passage/trec-dl-2019/judged
          inference_batch_size: 4

.. collapse:: recipe_dpr_search.py

   .. code-block:: python

      from lightning_ir import (
          BiEncoderModule, QueryDataset, SearchCallback,
          LightningIRDataModule, LightningIRTrainer,
          FaissSearchConfig,
      )

      module = BiEncoderModule(
          model_name_or_path="./my-dpr-checkpoint",
          evaluation_metrics=["nDCG@10"],
      )
      data_module = LightningIRDataModule(
          inference_datasets=[
              QueryDataset("msmarco-passage/trec-dl-2019/judged"),
          ],
          inference_batch_size=4,
      )
      callback = SearchCallback(
          index_dir="./msmarco-passage-index",
          search_config=FaissSearchConfig(k=100),
          save_dir="./runs",
      )
      trainer = LightningIRTrainer(
          callbacks=[callback], logger=False, enable_checkpointing=False
      )
      trainer.search(module, data_module)

**Step 4 — Re-rank with a cross-encoder:**

.. code-block:: bash

   lightning-ir re_rank --config recipe-dpr-rerank.yaml

.. collapse:: recipe-dpr-rerank.yaml

   .. code-block:: yaml

      trainer:
        logger: false
        callbacks:
          - class_path: lightning_ir.ReRankCallback
            init_args:
              save_dir: ./re-ranked-runs
      model:
        class_path: lightning_ir.CrossEncoderModule
        init_args:
          model_name_or_path: webis/monoelectra-base
          evaluation_metrics:
            - nDCG@10
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          inference_datasets:
            - class_path: lightning_ir.RunDataset
              init_args:
                run_path_or_id: ./runs/msmarco-passage-trec-dl-2019-judged.run
          inference_batch_size: 4

.. collapse:: recipe_dpr_rerank.py

   .. code-block:: python

      from lightning_ir import (
          CrossEncoderModule, RunDataset, ReRankCallback,
          LightningIRDataModule, LightningIRTrainer,
      )

      module = CrossEncoderModule(
          model_name_or_path="webis/monoelectra-base",
          evaluation_metrics=["nDCG@10"],
      )
      data_module = LightningIRDataModule(
          inference_datasets=[
              RunDataset("./runs/msmarco-passage-trec-dl-2019-judged.run"),
          ],
          inference_batch_size=4,
      )
      callback = ReRankCallback(save_dir="./re-ranked-runs")
      trainer = LightningIRTrainer(
          callbacks=[callback], logger=False, enable_checkpointing=False
      )
      trainer.re_rank(module, data_module)


Recipe 2: SPLADE Sparse Retrieval Pipeline
++++++++++++++++++++++++++++++++++++++++++

*Goal:* Train a SPLADE model with proper regularization, build a sparse index,
and retrieve.

**Step 1 — Fine-tune SPLADE with FLOPS regularization:**

.. code-block:: bash

   lightning-ir fit --config recipe-splade-fit.yaml

.. collapse:: recipe-splade-fit.yaml

   .. code-block:: yaml

      trainer:
        max_steps: 100_000
        precision: bf16-mixed
        callbacks:
          - class_path: lightning_ir.GenericConstantSchedulerWithLinearWarmup
            init_args:
              keys:
                - loss_functions.1.query_weight
                - loss_functions.1.doc_weight
              num_warmup_steps: 20_000
              num_delay_steps: 50_000
      model:
        class_path: lightning_ir.BiEncoderModule
        init_args:
          model_name_or_path: bert-base-uncased
          config:
            class_path: lightning_ir.models.SpladeConfig
          loss_functions:
            - lightning_ir.InBatchCrossEntropy
            - class_path: lightning_ir.FLOPSRegularization
              init_args:
                query_weight: 0.0008
                doc_weight: 0.0006
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          train_dataset:
            class_path: lightning_ir.TupleDataset
            init_args:
              tuples_dataset: msmarco-passage/train/triples-small
          train_batch_size: 32
      optimizer:
        class_path: torch.optim.AdamW
        init_args:
          lr: 1e-5

.. collapse:: recipe_splade_fit.py

   .. code-block:: python

      from torch.optim import AdamW
      from lightning_ir import (
          BiEncoderModule, LightningIRDataModule, LightningIRTrainer,
          TupleDataset, InBatchCrossEntropy, FLOPSRegularization,
          GenericConstantSchedulerWithLinearWarmup,
      )
      from lightning_ir.models import SpladeConfig

      module = BiEncoderModule(
          model_name_or_path="bert-base-uncased",
          config=SpladeConfig(),
          loss_functions=[
              InBatchCrossEntropy(),
              FLOPSRegularization(query_weight=0.0008, doc_weight=0.0006),
          ],
      )
      module.set_optimizer(AdamW, lr=1e-5)

      data_module = LightningIRDataModule(
          train_dataset=TupleDataset("msmarco-passage/train/triples-small"),
          train_batch_size=32,
      )
      scheduler = GenericConstantSchedulerWithLinearWarmup(
          keys=[
              "loss_functions.1.query_weight",
              "loss_functions.1.doc_weight",
          ],
          num_warmup_steps=20_000,
          num_delay_steps=50_000,
      )
      trainer = LightningIRTrainer(
          max_steps=100_000,
          precision="bf16-mixed",
          callbacks=[scheduler],
      )
      trainer.fit(module, data_module)

**Step 2 — Build a sparse index:**

.. code-block:: bash

   lightning-ir index --config recipe-splade-index.yaml

.. collapse:: recipe-splade-index.yaml

   .. code-block:: yaml

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
          model_name_or_path: ./my-splade-checkpoint
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          inference_datasets:
            - class_path: lightning_ir.DocDataset
              init_args:
                doc_dataset: msmarco-passage
          inference_batch_size: 256

.. collapse:: recipe_splade_index.py

   .. code-block:: python

      from lightning_ir import (
          BiEncoderModule, DocDataset, IndexCallback,
          LightningIRDataModule, LightningIRTrainer,
          TorchSparseIndexConfig,
      )

      module = BiEncoderModule(model_name_or_path="./my-splade-checkpoint")
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

**Step 3 — Sparse search:**

.. code-block:: bash

   lightning-ir search --config recipe-splade-search.yaml

.. collapse:: recipe-splade-search.yaml

   .. code-block:: yaml

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
          model_name_or_path: ./my-splade-checkpoint
          evaluation_metrics:
            - nDCG@10
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          inference_datasets:
            - class_path: lightning_ir.QueryDataset
              init_args:
                query_dataset: msmarco-passage/trec-dl-2019/judged
          inference_batch_size: 4

.. collapse:: recipe_splade_search.py

   .. code-block:: python

      from lightning_ir import (
          BiEncoderModule, QueryDataset, SearchCallback,
          LightningIRDataModule, LightningIRTrainer,
          TorchSparseSearchConfig,
      )

      module = BiEncoderModule(
          model_name_or_path="./my-splade-checkpoint",
          evaluation_metrics=["nDCG@10"],
      )
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


Recipe 3: ColBERT Multi-Vector Pipeline
+++++++++++++++++++++++++++++++++++++++

*Goal:* Fine-tune a ColBERT model, build a PLAID index for fast retrieval,
and search.

**Step 1 — Fine-tune ColBERT:**

.. code-block:: bash

   lightning-ir fit --config recipe-colbert-fit.yaml

.. collapse:: recipe-colbert-fit.yaml

   .. code-block:: yaml

      trainer:
        max_steps: 100_000
        precision: bf16-mixed
        accumulate_grad_batches: 4
        gradient_clip_val: 1
      model:
        class_path: lightning_ir.BiEncoderModule
        init_args:
          model_name_or_path: bert-base-uncased
          config:
            class_path: lightning_ir.models.ColConfig
            init_args:
              similarity_function: dot
              query_aggregation_function: sum
              query_expansion: true
              query_length: 32
              doc_length: 256
              normalization_strategy: l2
              embedding_dim: 128
              projection: linear_no_bias
              add_marker_tokens: true
          loss_functions:
            - lightning_ir.RankNet
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          train_dataset:
            class_path: lightning_ir.TupleDataset
            init_args:
              tuples_dataset: msmarco-passage/train/triples-small
          train_batch_size: 32
      optimizer:
        class_path: torch.optim.AdamW
        init_args:
          lr: 1e-5

.. collapse:: recipe_colbert_fit.py

   .. code-block:: python

      from torch.optim import AdamW
      from lightning_ir import (
          BiEncoderModule, LightningIRDataModule,
          LightningIRTrainer, RankNet, TupleDataset,
      )
      from lightning_ir.models import ColConfig

      module = BiEncoderModule(
          model_name_or_path="bert-base-uncased",
          config=ColConfig(
              similarity_function="dot",
              query_aggregation_function="sum",
              query_expansion=True,
              query_length=32,
              doc_length=256,
              normalization_strategy="l2",
              embedding_dim=128,
              projection="linear_no_bias",
              add_marker_tokens=True,
          ),
          loss_functions=[RankNet()],
      )
      module.set_optimizer(AdamW, lr=1e-5)

      data_module = LightningIRDataModule(
          train_dataset=TupleDataset("msmarco-passage/train/triples-small"),
          train_batch_size=32,
      )
      trainer = LightningIRTrainer(
          max_steps=100_000,
          precision="bf16-mixed",
          accumulate_grad_batches=4,
          gradient_clip_val=1,
      )
      trainer.fit(module, data_module)

**Step 2 — Build a PLAID index:**

.. code-block:: bash

   lightning-ir index --config recipe-colbert-index.yaml

.. collapse:: recipe-colbert-index.yaml

   .. code-block:: yaml

      trainer:
        logger: false
        callbacks:
          - class_path: lightning_ir.IndexCallback
            init_args:
              index_dir: ./colbert-index
              index_config:
                class_path: lightning_ir.PlaidIndexConfig
      model:
        class_path: lightning_ir.BiEncoderModule
        init_args:
          model_name_or_path: ./my-colbert-checkpoint
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          inference_datasets:
            - class_path: lightning_ir.DocDataset
              init_args:
                doc_dataset: msmarco-passage
          inference_batch_size: 256

.. collapse:: recipe_colbert_index.py

   .. code-block:: python

      from lightning_ir import (
          BiEncoderModule, DocDataset, IndexCallback,
          LightningIRDataModule, LightningIRTrainer,
          PlaidIndexConfig,
      )

      module = BiEncoderModule(model_name_or_path="./my-colbert-checkpoint")
      data_module = LightningIRDataModule(
          inference_datasets=[DocDataset("msmarco-passage")],
          inference_batch_size=256,
      )
      callback = IndexCallback(
          index_dir="./colbert-index",
          index_config=PlaidIndexConfig(),
      )
      trainer = LightningIRTrainer(
          callbacks=[callback], logger=False, enable_checkpointing=False
      )
      trainer.index(module, data_module)

**Step 3 — Search with PLAID:**

.. code-block:: bash

   lightning-ir search --config recipe-colbert-search.yaml

.. collapse:: recipe-colbert-search.yaml

   .. code-block:: yaml

      trainer:
        logger: false
        callbacks:
          - class_path: lightning_ir.SearchCallback
            init_args:
              index_dir: ./colbert-index
              search_config:
                class_path: lightning_ir.PlaidSearchConfig
                init_args:
                  k: 100
              save_dir: ./runs
      model:
        class_path: lightning_ir.BiEncoderModule
        init_args:
          model_name_or_path: ./my-colbert-checkpoint
          evaluation_metrics:
            - nDCG@10
      data:
        class_path: lightning_ir.LightningIRDataModule
        init_args:
          inference_datasets:
            - class_path: lightning_ir.QueryDataset
              init_args:
                query_dataset: msmarco-passage/trec-dl-2019/judged
          inference_batch_size: 4

.. collapse:: recipe_colbert_search.py

   .. code-block:: python

      from lightning_ir import (
          BiEncoderModule, QueryDataset, SearchCallback,
          LightningIRDataModule, LightningIRTrainer,
          PlaidSearchConfig,
      )

      module = BiEncoderModule(
          model_name_or_path="./my-colbert-checkpoint",
          evaluation_metrics=["nDCG@10"],
      )
      data_module = LightningIRDataModule(
          inference_datasets=[
              QueryDataset("msmarco-passage/trec-dl-2019/judged"),
          ],
          inference_batch_size=4,
      )
      callback = SearchCallback(
          index_dir="./colbert-index",
          search_config=PlaidSearchConfig(k=100),
          save_dir="./runs",
      )
      trainer = LightningIRTrainer(
          callbacks=[callback], logger=False, enable_checkpointing=False
      )
      trainer.search(module, data_module)


---------------------------------
Quick Reference: Compatibility
---------------------------------

Use this table as a cheat sheet when composing configurations.

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 18 28

   * - Model Config
     - Module
     - Compatible Index
     - Compatible Search
     - Supported Workflows
   * - :py:class:`~lightning_ir.models.bi_encoders.dpr.DprConfig`
     - :py:class:`~lightning_ir.bi_encoder.bi_encoder_module.BiEncoderModule`
     - :py:class:`~lightning_ir.retrieve.pytorch.dense_indexer.TorchDenseIndexConfig`, :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissFlatIndexConfig`,
       :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissIVFIndexConfig`, :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissIVFPQIndexConfig`
     - :py:class:`~lightning_ir.retrieve.pytorch.dense_searcher.TorchDenseSearchConfig`, :py:class:`~lightning_ir.retrieve.faiss.faiss_searcher.FaissSearchConfig`
     - fit, index, search, re_rank
   * - :py:class:`~lightning_ir.models.bi_encoders.splade.SpladeConfig`
     - :py:class:`~lightning_ir.bi_encoder.bi_encoder_module.BiEncoderModule`
     - :py:class:`~lightning_ir.retrieve.pytorch.sparse_indexer.TorchSparseIndexConfig`, :py:class:`~lightning_ir.retrieve.seismic.seismic_indexer.SeismicIndexConfig`
     - :py:class:`~lightning_ir.retrieve.pytorch.sparse_searcher.TorchSparseSearchConfig`, :py:class:`~lightning_ir.retrieve.seismic.seismic_searcher.SeismicSearchConfig`
     - fit, index, search, re_rank
   * - :py:class:`~lightning_ir.models.bi_encoders.col.ColConfig`
     - :py:class:`~lightning_ir.bi_encoder.bi_encoder_module.BiEncoderModule`
     - :py:class:`~lightning_ir.retrieve.pytorch.dense_indexer.TorchDenseIndexConfig`, :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissFlatIndexConfig`,
       :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissIVFIndexConfig`, :py:class:`~lightning_ir.retrieve.faiss.faiss_indexer.FaissIVFPQIndexConfig`,
       :py:class:`~lightning_ir.retrieve.plaid.plaid_indexer.PlaidIndexConfig`
     - :py:class:`~lightning_ir.retrieve.pytorch.dense_searcher.TorchDenseSearchConfig`, :py:class:`~lightning_ir.retrieve.faiss.faiss_searcher.FaissSearchConfig`,
       :py:class:`~lightning_ir.retrieve.plaid.plaid_searcher.PlaidSearchConfig`
     - fit, index, search, re_rank
   * - :py:class:`~lightning_ir.models.cross_encoders.mono.MonoConfig`
     - :py:class:`~lightning_ir.cross_encoder.cross_encoder_module.CrossEncoderModule`
     - —
     - —
     - fit, re_rank
   * - :py:class:`~lightning_ir.models.cross_encoders.set_encoder.SetEncoderConfig`
     - :py:class:`~lightning_ir.cross_encoder.cross_encoder_module.CrossEncoderModule`
     - —
     - —
     - fit, re_rank
