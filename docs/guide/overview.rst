.. _guide-overview:

=========================
What Do You Want to Do?
=========================

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

Continue with the next decision:

- :doc:`models` — Pick a model architecture
- :doc:`index-types` — Pick an index type (bi-encoder only)
- :doc:`loss-functions` — Pick a loss function for training
- :doc:`datasets` — Pick a dataset format
- :doc:`recipes` — Jump straight to a complete end-to-end recipe
