.. _quickstart:

================
Quickstart Guide
================

Lightning IR can either be used programatically or using the CLI. The CLI is based on `PyTorch Lightning CLI <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli>`_ and adds additional options to provide a unified interface for fine-tuning and running neural ranking models.

After installing Lightning IR, the CLI is accessible via the ``lightning-ir`` command and provides commands for fine-tuning, indexing, searching, and re-ranking. 

.. code-block::

    $ lightning-ir --help
    
    ...
    
    Available subcommands:
      fit                 Runs the full optimization routine.
      index               Index a collection of documents.
      search              Search for relevant documents.
      re_rank             Re-rank a set of retrieved documents.

The behavior of the CLI is most easily controlled using YAML configuration files which specify the model, data, and trainer settings.

Example
-------

The following sections provide a step-by-step example of how to fine-tune a bi-encoder or a cross-encoder model on the MS MARCO passage ranking dataset, index the documents, search for relevant documents, and re-rank these documents. If you are only interested in inference and not fine-tuning, you can skip the :ref:`fine-tuning` step and directly jump to the :ref:`indexing`, :ref:`searching`, or :ref:`re-ranking` steps using already fine-tuned models from the :ref:`model-zoo`.

.. _fine-tuning:

Fine-Tuning
+++++++++++

To fine-tune a model you need to define the model module (either a :py:class:`~lightning_ir.bi_encoder.module.BiEncoderModule` or a :py:class:`~lightning_ir.cross_encoder.module.CrossEncoderModule`), the :py:class:`~lightning_ir.data.datamodule.LightningIRDataModule` (which has either a :py:class:`~lightning_ir.data.dataset.TupleDataset` or :py:class:`~lightning_ir.data.dataset.RunDataset` training dataset), and the :py:class:`~lightning_ir.main.LightningIRTrainer` settings.   

The following command and configuration file demonstrates how to fine-tune a bi-encoder (or cross-encoder) on the MS MARCO passage ranking dataset using the CLI.

.. code-block:: bash
  
      lightning-ir fit --config fine-tune.yaml

.. collapse:: fine-tune.yaml

  .. literalinclude:: ../configs/examples/fine-tune.yaml
    :language: yaml

The following script demonstrates how to do the same but programatically.

.. collapse:: fine_tune.py

  .. literalinclude:: ../examples/fine_tune.py
    :language: python

.. _indexing:

Indexing
++++++++

For indexing, you need an already fine-tuned :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel`. See the :ref:`model-zoo` for examples. Depending on the bi-encoder model type, you need to select the appropriate :py:class:`~lightning_ir.retrieve.indexer.IndexConfig` to pass to the :py:class:`~lightning_ir.lightning_utils.callbacks.IndexCallback`. In addition, you need to specify the :py:class:`~lightning_ir.data.dataset.DocDataset` to index. The model module, data module, and indexing callback are then passed to the trainer to run the indexing.

The following command and configuration file demonstrate how to index the MS MARCO passage ranking dataset using an already fine-tuned bi-encoder and `faiss <https://faiss.ai/>`_.

.. code-block:: bash
  
      lightning-ir index --config index.yaml

.. collapse:: index.yaml

  .. literalinclude:: ../configs/examples/index.yaml
    :language: yaml

The following script demonstrates how to do the same but programatically.

.. collapse:: index.py

  .. literalinclude:: ../examples/index.py
    :language: python

.. _searching:

Searching
+++++++++

For searching, you need an already fine-tuned :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel`. See the :ref:`model-zoo` for examples. Additionally, you must have created an index using the :ref:`indexing` step. The search is performed using the :py:class:`~lightning_ir.lightning_utils.callbacks.SearchCallback` which requires a :py:class:`~lightning_ir.retrieve.searcher.SearchConfig` that corresponds to the :py:class:`~lightning_ir.retrieve.indexer.IndexConfig` used during indexing. The data module must receive a :py:class:`~lightning_ir.data.dataset.QueryDataset` to iterate over a set of queries. The model module, data module, and searching callback are then passed to the trainer to run searching. If the dataset has relevance judgements and a set of evaluation metrics are passed to the model, the trainer will report effectiveness metrics.

The following command and configuration file demonstrate how to retrieve the top-100 passages for each query from the TREC Deep Learning 2019 and 2020 tracks. After searching, the results are saved in a run file and the effectiveness is reported using nDCG\@10.

.. code-block:: bash
  
      lightning-ir search --config search.yaml

.. collapse:: search.yaml

  .. literalinclude:: ../configs/examples/search.yaml

The following script demonstrates how to do the same but programatically.

.. collapse:: search.py

  .. literalinclude:: ../examples/search.py

.. _re-ranking:

Re-Ranking
++++++++++

For re-ranking, you need an already fine-tuned :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel` or :py:class:`~lightning_ir.cross_encoder.model.CrossEncoderModel` (the latter are usually more effective). The data module must receive a :py:class:`~lightning_ir.data.dataset.RunDataset` which loads the run file to re-rank. To save the re-ranked file you need to specify a :py:class:`~lightning_ir.lightning_utils.callbacks.ReRankCallback`. The model module, data module, and re-ranking callback are then passed to the trainer to run re-ranking. If the dataset has relevance judgements and a set of evaluation metrics are passed to the model, the trainer will report effectiveness metrics.

The following command and configuration file demonstrate how to re-rank the top-100 passages for each query from the TREC Deep Learning 2019 and 2020 tracks using a cross-encoder. After re-ranking, the results are saved in a run file and the effectiveness is reported using nDCG\@10.

.. code-block:: bash
  
      lightning-ir re_rank --config re-rank.yaml

.. collapse:: re-rank.yaml

  .. literalinclude:: ../configs/examples/re-rank.yaml

The following script demonstrates how to do the same but programatically.

.. collapse:: re_rank.py

  .. literalinclude:: ../examples/re_rank.py