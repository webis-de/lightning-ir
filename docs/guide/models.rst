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

DPR encodes each query and document into a single dense vector using a
pooling step (typically the CLS token). Similarity is computed with a single
dot product or cosine comparison, which makes retrieval fast and straightforward
to index. The optional ``projection`` head lets you reduce the embedding
dimension and add a non-linear bottleneck; set ``embedding_dim`` to control the
output size.

.. code-block:: yaml

   # model-dpr.yaml
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: bert-base-uncased
       config:
         class_path: lightning_ir.models.DprConfig
         init_args:
           similarity_function: dot   # or "cosine"
           query_length: 32           # max query tokens
           doc_length: 512            # max document tokens
           pooling_strategy: first    # CLS token; also "mean", "max", "sum"
           embedding_dim: null        # null → use backbone hidden size
           projection: linear         # linear projection head; null to disable

.. code-block:: python

   from lightning_ir import BiEncoderModule
   from lightning_ir.models import DprConfig

   module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=DprConfig(
           similarity_function="dot",
           query_length=32,
           doc_length=512,
           pooling_strategy="first",
           embedding_dim=None,   # None → use backbone hidden size
           projection="linear",
       ),
   )

**ColBERT** — multi-vector late interaction:

ColBERT produces one embedding **per token**, so controlling sequence lengths and
the per-token embedding dimension directly affects index size and quality.
It also supports special ColBERT-specific options (query expansion, marker
tokens, per-token ``normalization_strategy``) that DPR and SPLADE do not have.

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
           query_expansion: true         # ColBERT-specific: pad queries to query_length
           query_length: 32
           doc_length: 256               # kept smaller than DPR to limit index size
           normalization_strategy: l2   # ColBERT-specific: per-token l2 normalisation
           embedding_dim: 128            # project every token to 128-d (reduces index size)
           projection: linear_no_bias   # ColBERT convention: no bias in projection
           add_marker_tokens: true       # ColBERT-specific: [Q]/[D] special tokens

.. code-block:: python

   from lightning_ir import BiEncoderModule
   from lightning_ir.models import ColConfig

   module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=ColConfig(
           similarity_function="dot",
           query_aggregation_function="sum",
           query_expansion=True,         # ColBERT-specific
           query_length=32,
           doc_length=256,
           normalization_strategy="l2",  # ColBERT-specific
           embedding_dim=128,
           projection="linear_no_bias",
           add_marker_tokens=True,       # ColBERT-specific
       ),
   )

**SPLADE** — learned sparse retrieval:

SPLADE maps each query and document to a sparse vector over the full vocabulary.
Each vocabulary dimension is activated by taking the max-pool of the token
logits, so the representation is directly interpretable as a bag of weighted
terms. Because the embedding space is the tokenizer vocabulary, ``embedding_dim``
cannot be set freely — it is always equal to the vocabulary size. The key
knob is ``pooling_strategy``: ``max`` (the default) gives standard SPLADE
behaviour; ``sum`` gives SPLADE-doc behaviour.

.. code-block:: yaml

   # model-splade.yaml
   model:
     class_path: lightning_ir.BiEncoderModule
     init_args:
       model_name_or_path: bert-base-uncased
       config:
         class_path: lightning_ir.models.SpladeConfig
         init_args:
           similarity_function: dot   # sparse dot product
           query_length: 32           # max query tokens
           doc_length: 512            # max document tokens
           pooling_strategy: max      # max over token activations (SPLADE default)

.. note::

   SPLADE uses the full vocabulary as its embedding space, so
   ``embedding_dim`` is tied to the vocabulary size and cannot be configured
   via the constructor (it is derived from the backbone tokenizer).

.. code-block:: python

   from lightning_ir import BiEncoderModule
   from lightning_ir.models import SpladeConfig

   module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=SpladeConfig(
           similarity_function="dot",
           query_length=32,
           doc_length=512,
           pooling_strategy="max",  # max over token activations (SPLADE default)
       ),
   )

**Cross-encoder (MonoEncoder)** — highest quality re-ranking:

A cross-encoder concatenates query and document into a single input and runs
them jointly through the backbone, so every layer can attend across both texts.
This yields the highest scoring quality but means no pre-indexing is possible
— documents must be re-encoded for every new query. Use this architecture when
you already have a candidate list (e.g. from a bi-encoder first stage) and
need the best possible re-ranking without latency constraints.

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
