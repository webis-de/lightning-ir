.. _guide-models:

================================
Which Model Architecture to Use?
================================

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
-----------------------

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
   * - **MVR**
     - :py:class:`~lightning_ir.models.bi_encoders.mvr.MvrConfig`
     - Separate
     - Multi dense (viewer tokens)
     - ✅
     - ✅
     - Fixed-count multi-vector; balances index size and quality
   * - **COIL**
     - :py:class:`~lightning_ir.models.bi_encoders.coil.CoilConfig`
     - Separate
     - Multi (token + CLS)
     - ✅
     - ✅
     - Exact lexical match with context; sparse + dense hybrid
   * - **UniCOIL**
     - :py:class:`~lightning_ir.models.bi_encoders.coil.UniCoilConfig`
     - Separate
     - Single sparse (token weights)
     - ✅
     - ✅
     - Lightweight token-weight sparse retrieval; simpler than SPLADE

.. note::

   External checkpoints are also supported out of the box. For example, XTR
   (``google/xtr-base-en``) uses the ColBERT architecture (:py:class:`~lightning_ir.models.bi_encoders.col.ColConfig`)
   with a T5 backbone. See the :ref:`concepts-model` page and the :py:mod:`~lightning_ir.models` API
   reference for a full list of registered checkpoints.

Quick Examples
--------------

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

Next steps:

- :doc:`index-types` — Choose an index type for your bi-encoder
- :doc:`loss-functions` — Choose a loss function for training
- :doc:`recipes` — See a complete end-to-end pipeline
