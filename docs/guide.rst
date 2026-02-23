.. _guide:

==============
Decision Guide
==============

This guide helps you navigate Lightning IR's configuration space. It is structured
as a series of decision trees: start with *what you want to do*, then follow the
branches to pick the right model architecture, index type, loss function, and data
format. Each section ends with concrete CLI and Python examples you can copy and
adapt.

.. toctree::
   :maxdepth: 1
   :hidden:

   guide/overview
   guide/models
   guide/index-types
   guide/loss-functions
   guide/datasets
   guide/recipes

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What you will find
   * - :doc:`guide/overview`
     - The four top-level workflows (``fit``, ``index``, ``search``, ``re_rank``)
       and a summary table of the required ingredients for each.
   * - :doc:`guide/models`
     - Decision tree and comparison table for choosing a model architecture
       (DPR, SPLADE, ColBERT, MonoEncoder, SetEncoder), with quick-start code examples.
   * - :doc:`guide/index-types`
     - Decision tree and comparison table for choosing an index and search config
       (FAISS variants, PLAID, Torch dense/sparse, Seismic), with quick-start code examples.
   * - :doc:`guide/loss-functions`
     - Decision tree and reference table for choosing a loss function, including
       knowledge-distillation and SPLADE regularization examples.
   * - :doc:`guide/datasets`
     - Decision tree and reference table for the four dataset classes
       (TupleDataset, RunDataset, DocDataset, QueryDataset).
   * - :doc:`guide/recipes`
     - Complete end-to-end pipelines for DPR, SPLADE, and ColBERT — each covering
       fine-tuning, indexing, searching, and re-ranking. Includes a compatibility
       cheat sheet.
