.. _concepts-trainer:

=======
Trainer
=======

The :py:class:`~lightning_ir.main.LightningIRTrainer` derives from a `PyTorch Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ to enable easy, scalable, and reproducible fine-tuning. It furthermore adds functionality for additional information retrieval stages—namely, indexing, searching, and re-ranking. The trainer combines a :py:class:`~lightning_ir.base.module.LightningIRModule` and a :py:class:`~lightning_ir.data.datamodule.LightningIRDataModule` and handles the fine-tuning and inference logic.

The following sections provide an overview of the trainer's functionality for the different stages and how to use it.

.. note:: 
    Lightning IR provides an easy-to-use CLI (based off the `Pytorch Lightning CLI <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli>`_) that wraps the trainer and provides commands for all the retrieval stages exemplified below. See the :ref:`quickstart` for usage examples.

Fine-Tuning
-----------

The :py:class:`~lightning_ir.main.LightningIRTrainer` is designed for scalable and efficient fine-tuning of models. See the documentation of the parent `PyTorch Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ for details on hyperparameters, parallelization, logging, checkpointing, and more.

The ``fit`` method of the :py:class:`~lightning_ir.main.LightningIRTrainer` starts the fine-tuning process. A :py:class:`~lightning_ir.base.module.LightningIRModule`, a :py:class:`~lightning_ir.data.datamodule.LightningIRDataModule`, an `Optimizer <https://pytorch.org/docs/stable/optim.html>`_, and at least one :py:class:`~lightning_ir.loss.loss.LossFunction` must first be configured. Several popular loss functions for fine-tuning ranking models are available in the :py:mod:`~lightning_ir.loss.loss` module. The snippet below demonstrates how to fine-tune a :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` or a :py:class:`~lightning_ir.cross_encoder.model.CrossEncoderModel` on the official `MS MARCO <https://microsoft.github.io/msmarco/>`_ triples.

.. literalinclude:: ../../examples/fine_tune.py
    :language: python

Indexing
--------

To index a document collection using an already fine-tuned :py:class:`lightning_ir.bi_encoder.model.BiEncoderModel` use the ``index`` method of the :py:class:`~lightning_ir.main.LightningIRTrainer`. The trainer must receive :py:class:`~lightning_ir.lightning_utils.callbacks.IndexCallback` which handles writing the document embeddings to disk. The :py:class:`~lightning_ir.lightning_utils.callbacks.IndexCallback` is configured with a :py:class:`~lightning_ir.retrieve.indexer.IndexConfig` that specifies the type of index to use and how this index should be configured. If the selected :py:class:`lightning_ir.bi_encoder.model.BiEncoderModel` generates sparse embeddings, a :py:class:`~lightning_ir.retrieve.indexer.SparseIndexConfig` should be used. For dense embeddings, Lightning IR provides a :py:class:`lightning_ir.retrieve.faiss_indexer.FaissIndexConfig` that uses `faiss <https://faiss.ai/>`_ for fast approximate nearest neighbor search. The snippet below demonstrates how to index the MS MARCO passage ranking dataset using an already fine-tuned bi-encoder.

.. literalinclude:: ../../examples/index.py
    :language: python

Searching
---------

To search for relevant documents given a query using an already fine-tuned :py:class:`lightning_ir.bi_encoder.model.BiEncoderModel`, use the ``search`` method of the :py:class:`~lightning_ir.main.LightningIRTrainer`. The trainer must receive a :py:class:`~lightning_ir.lightning_utils.callbacks.SearchCallback` which handles loading index and searching for relevant documents based on the generated query embeddings. The :py:class:`~lightning_ir.lightning_utils.callbacks.SearchCallback` is configured with a :py:class:`~lightning_ir.retrieve.searcher.SearchConfig` that must match the index configuration used during indexing. To save the results to disk in the form of a run file, you can optionally add a :py:class:`lightning_ir.lightning_utils.callbacks.RankCallback` and specify a directory to save run files to. The snippet below demonstrates how to search for relevant documents given a query using an already fine-tuned bi-encoder.

.. literalinclude:: ../../examples/search.py
    :language: python

Re-Ranking
----------

To re-rank a set of retrieved documents using an already fine-tuned :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` or :py:class:`~lightning_ir.cross_encoder.model.CrossEncoderModel` (the latter are usually more effective), use the ``re_rank`` method of the :py:class:`~lightning_ir.main.LightningIRTrainer`. The trainer must receive a :py:class:`~lightning_ir.lightning_utils.callbacks.ReRankCallback` which handles saving the re-ranked run file to disk. The snippet below demonstrates how to re-rank a set of retrieved documents using an already fine-tuned cross-encoder.

.. literalinclude:: ../../examples/re_rank.py
    :language: python