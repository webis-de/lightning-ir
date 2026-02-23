.. _guide-loss-functions:

===========================
Which Loss Function to Use?
===========================

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
-----------------------

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
------------------------------------------

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
--------------------------------------

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

Next steps:

- :doc:`datasets` — Choose the right dataset format
- :doc:`recipes` — See complete end-to-end pipelines
