=============
Core Concepts
=============

.. toctree:: 
    :maxdepth: 1
    :hidden:

    concepts/model
    concepts/data
    concepts/trainer

Lightning IR is built on top of `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_ and shares most of its core concepts and API. Concretely, the :py:class:`~lightning_ir.base.module.LightningIRModule` handles the model, the :py:class:`~lightning_ir.data.datamodule.LightningIRDataModule` handles the data, and the :py:class:`~lightning_ir.main.LightningIRTrainer` handles combining the model and data. The :ref:`concepts-model`, :ref:`concepts-data`, and :ref:`concepts-trainer` sections provide more details on these core concepts.