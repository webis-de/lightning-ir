==========
Quickstart
==========

The easiest way to use Lightning IR is via the CLI. It uses the `PyTorch Lightning CLI <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli>`_ and adds additional options to provide a unified interface for fine-tuning and running neural ranking models.

After installing Lightning IR, the CLI is accessible via the ``lightning-ir`` command and provides commands for fine-tuning, indexing, searching, and re-ranking. 

.. code-block::

    $ lightning-ir --help
    
    ...
    
    Available subcommands:
      fit                 Runs the full optimization routine.
      index               Index a collection of documents.
      search              Search for relevant documents.
      re_rank             Re-rank a set of retrieved documents.

The behavior of the CLI is most easily controlled using YAML configuration files which specify the model, data, and trainer settings. The following sections provide examples for each of the available commands.

Fine-Tuning
-----------


Indexing
--------


Searching
---------


Re-Ranking
----------