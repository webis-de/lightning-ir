.. _howto-model:

====================
Build A Custom Model
====================

This section provides step-by-step guides on how to build custom :py:class:`~lightning_ir.bi_encoder.bi_encoder_model.BiEncoderModel` and :py:class:`~lightning_ir.cross_encoder.cross_encoder_model.CrossEncoderModel` models in Lightning IR.

Bi-Encoder
----------

Say we wanted to build a custom bi-encoder model that adds an additional linear layer on top of the pooled embeddings. If we wanted to make this option configurable, we would first need to subclass the :py:class:`~lightning_ir.bi_encoder.bi_encoder_config.BiEncoderConfig` and add a new attribute for the additional linear layer. We must also assign a new ``model_type`` to our model. For example:

.. code-block:: python

    from lightning_ir.bi_encoder.config import BiEncoderConfig

    class CustomBiEncoderConfig(BiEncoderConfig):
        model_type = "custom-bi-encoder"

        def __init__(self, additional_linear_layer = True, **kwargs):
            super().__init__(**kwargs)
            self.additional_linear_layer = additional_linear_layer

Next, we need to subclass the :py:class:`~lightning_ir.bi_encoder.bi_encoder_model.BiEncoderModel` and override the :py:class:`~lightning_ir.bi_encoder.bi_encoder_model.BiEncoderModel.encode` method to include the additional linear layer. We also need to ensure that our new config class is registered with our new model as the :py:meth:`~lightning_ir.bi_encoder.bi_encoder_model.BiEncoderModel.config_class` attribute. In the :py:class:`~lightning_ir.bi_encoder.model.BiEncoderModel.encode` method, the :py:meth:`~lightning_ir.bi_encoder.bi_encoder_model.BiEncoderModel._backbone_forward` method runs the backbone model and returns the contextualized embeddings of the input sequence. We then apply our additional linear layer to the pooled embeddings. Afterwards, the various steps of the processing pipeline for bi-encoders are applied (see :ref:`concepts-model` for more details). For example:

.. code-block:: python
    
    from typing import Literal

    import torch
    from transformers import BatchEncoding

    from lightning_ir import BiEncoderModel, BiEncoderOutput

    class CustomBiEncoderModel(BiEncoderModel):
        config_class = CustomBiEncoderConfig

        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self.additional_linear_layer = None
            if config.additional_linear_layer:
                self.additional_linear_layer = torch.nn.Linear(
                    config.hidden_size, config.hidden_size
                )

        def encode(
            self,
            encoding: BatchEncoding,
            expansion: bool = False,
            pooling_strategy: Literal["first", "mean", "max", "sum"] | None = None,
            mask_scoring_input_ids: torch.Tensor | None = None,
        ) -> BiEncoderEmbedding:
            embeddings = self._backbone_forward(**encoding).last_hidden_state
            if self.additional_linear_layer is not None:  # apply additional linear layer
                embeddings = self.additional_linear_layer(embeddings)
            if self.projection is not None:
                embeddings = self.projection(embeddings)
            embeddings = self._sparsification(embeddings, self.config.sparsification)
            embeddings = self._pooling(embeddings, encoding["attention_mask"], pooling_strategy)
            if self.config.normalization == "l2":
                embeddings = torch.nn.functional.normalization(embeddings, dim=-1)
            scoring_mask = self.scoring_mask(
                encoding["input_ids"],
                encoding["attention_mask"],
                expansion,
                pooling_strategy,
                mask_scoring_input_ids,
            )
            return BiEncoderEmbedding(embeddings, scoring_mask)

Finally, to make sure we can use our new model within the Hugging Face ecosystem, we need to register our model with the Hugging Face auto loading mechanism. We additionally need to register the :py:class:`~lightning_ir.bi_encoder.bi_encoder_tokenizer.BiEncoderTokenizer` to ensure it is loaded when loading our new model. We can do this by adding the following code to our model file:

.. code-block:: python

    from lightning_ir import BiEncoderTokenizer
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    AutoConfig.register(CustomBiEncoderConfig.model_type, CustomBiEncoderConfig)
    AutoModel.register(CustomBiEncoderConfig, CustomBiEncoderModel)
    AutoTokenizer.register(CustomBiEncoderConfig, BiEncoderTokenizer)

Now we can use our custom cross-encoder model in the same way as the built-in models. For example, to fine-tune our custom bi-encoder model on the MS MARCO dataset, we can use the following code:

.. code-block:: python

    from torch.optim import AdamW

    from lightning_ir import (
        BiEncoderModule,
        LightningIRDataModule,
        LightningIRTrainer,
        RankNet,
        TupleDataset,
    )

    module = BiEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=CustomBiEncoderConfig(), # our custom config
       loss_functions=[RankNet()]
    )
    module.set_optimizer(AdamW, lr=1e-5)
    data_module = LightningIRDataModule(
        train_dataset=TupleDataset("msmarco-passage/train/triples-small"),
        train_batch_size=32,
    )
    trainer = LightningIRTrainer(max_steps=100_000)
    trainer.fit(module, data_module)

Here is the full code for our custom bi-encoder model:

.. collapse:: custom_bi_encoder.py
    
    .. literalinclude:: ../../examples/custom_bi_encoder.py


Cross-Encoder
-------------

Say we wanted to build a custom cross-encoder model that adds an additional linear layer on top of the pooled embeddings. If we wanted to make this option configurable, we would first need to subclass the :py:class:`~lightning_ir.cross_encoder.cross_encoder_config.CrossEncoderConfig` and add a new attribute for the additional linear layer. We must also assign a new ``model_type`` to our model. For example:

.. code-block:: python

    from lightning_ir import CrossEncoderConfig

    class CustomCrossEncoderConfig(CrossEncoderConfig):
        model_type = "custom-cross-encoder"

        def __init__(self, additional_linear_layer = True, **kwargs):
            super().__init__(**kwargs)
            self.additional_linear_layer = additional_linear_layer

Next, we need to subclass the :py:class:`~lightning_ir.cross_encoder.cross_encoder_model.CrossEncoderModel` and override the :py:meth:`~lightning_ir.cross_encoder.cross_encoder_model.CrossEncoderModel.forward` method to include the additional linear layer. We also need to ensure that our new config class is registered with our new model as the :py:attr:`~lightning_ir.cross_encoder.cross_encoder_model.CrossEncoderModel.config_class` attribute. In the :py:meth:`~lightning_ir.cross_encoder.cross_encoder_model.CrossEncoderModel.forward` method, the :py:meth:`~lightning_ir.cross_encoder.cross_encoder_model.CrossEncoderModel._backbone_forward` method runs the backbone model and returns the contextualized embeddings of the input sequence. The :py:meth:`~lightning_ir.cross_encoder.cross_encoder_model.CrossEncoderModel._pooling` method aggregates the embeddings based on the pooling strategy specified in the config. We then apply our additional linear layer to the pooled embeddings and finally use a linear layer to compute the final relevance score. For example:

.. code-block:: python
    
    import torch
    from transformers import BatchEncoding

    from lightning_ir import CrossEncoderModel, CrossEncoderOutput


    class CustomCrossEncoderModel(CrossEncoderModel):
        config_class = CustomCrossEncoderConfig

        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self.additional_linear_layer = None
            if config.additional_linear_layer:
                self.additional_linear_layer = torch.nn.Linear(
                    config.hidden_size, config.hidden_size
                )

        def forward(self, encoding: BatchEncoding) -> torch.Tensor:
            embeddings = self._backbone_forward(**encoding).last_hidden_state
            embeddings = self._pooling(
                embeddings,
                encoding.get("attention_mask", None),
                pooling_strategy=self.config.pooling_strategy,
            )
            if self.additional_linear_layer is not None:
                embeddings = self.additional_linear_layer(embeddings)
            scores = self.linear(embeddings).view(-1)
            return CrossEncoderOutput(scores=scores, embeddings=embeddings)



Finally, to make sure we can use our new model within the Hugging Face ecosystem, we need to register our model with the Hugging Face auto loading mechanism. We additionally need to register the :py:class:`~lightning_ir.cross_encoder.cross_encoder_tokenizer.CrossEncoderTokenizer` to ensure it is loaded when loading our new model. We can do this by adding the following code to our model file:

.. code-block:: python

    from lightning_ir import CrossEncoderTokenizer
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    AutoConfig.register(CustomCrossEncoderConfig.model_type, CustomCrossEncoderConfig)
    AutoModel.register(CustomCrossEncoderConfig, CustomCrossEncoderModel)
    AutoTokenizer.register(CustomCrossEncoderConfig, CrossEncoderTokenizer)

Now we can use our custom cross-encoder model in the same way as the built-in models. For example, to fine-tune our custom cross-encoder model on the MS MARCO dataset, we can use the following code:

.. code-block:: python

    from torch.optim import AdamW

    from lightning_ir import (
        CrossEncoderModule,
        LightningIRDataModule,
        LightningIRTrainer,
        RankNet,
        TupleDataset,
    )

    module = CrossEncoderModule(
       model_name_or_path="bert-base-uncased",
       config=CustomCrossEncoderConfig(), # our custom config
       loss_functions=[RankNet()]
    )
    module.set_optimizer(AdamW, lr=1e-5)
    data_module = LightningIRDataModule(
        train_dataset=TupleDataset("msmarco-passage/train/triples-small"),
        train_batch_size=32,
    )
    trainer = LightningIRTrainer(max_steps=100_000)
    trainer.fit(module, data_module)

Here is the full code for our custom cross-encoder model:

.. collapse:: custom_cross_encoder.py
    
    .. literalinclude:: ../../examples/custom_cross_encoder.py
