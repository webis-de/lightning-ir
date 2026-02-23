.. _guide-recipes:

===================
End-to-End Recipes
===================

The following recipes chain together the decisions from the other guide pages
into complete, copy-pasteable pipelines. Each recipe shows both the CLI (YAML)
and programmatic (Python) approach.

.. tip::

   Not sure which recipe to use? Start with the :doc:`overview` to identify
   your workflow, then return here for the full pipeline.

Recipe 1: DPR Dense Retrieval Pipeline
---------------------------------------

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
-------------------------------------------

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
----------------------------------------

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


Quick Reference: Compatibility
-------------------------------

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
