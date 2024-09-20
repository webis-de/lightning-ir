.. _model-zoo:

=========
Model Zoo
=========

The following table lists models from the `HuggingFace Model Hub <https://huggingface.co/models>`_ that are supported in Lightning IR. For each model, the table reports the re-ranking effectiveness in terms of nDCG\@10 on the officially released run files containing 1,000 passages for TREC Deep Learning 2019 and 2020. 

Native models were fine-tuned using Lightning IR and the model's HuggingFace model card provides Lightning IR configurations for reproduction. Non-native models were fine-tuned externally but are supported in Lightning IR for inference.

**Reproduction**

The following command and configuration can be used to reproduce the results:

.. collapse:: config.yaml
    
    .. code-block:: yaml

        trainer:
          logger: false
        model:
          class_path: CrossEncoderModule # for cross-encoders
          # class_path: BiEncoderModule # for bi-encoders
          init_args:
            model_name_or_path: {MODEL_NAME}
            evaluation_metrics: 
            - nDCG@10
        data:
          class_path: LightningIRDataModule
          init_args:
            inference_datasets:
            - class_path: RunDataset
              init_args:
              run_path_or_id: msmarco-passage/trec-dl-2019/judged
            - class_path: RunDataset
              init_args:
              run_path_or_id: msmarco-passage/trec-dl-2020/judged

.. code-block:: bash

    lightning-ir re_rank --config config.yaml


.. csv-table::
    :file: ./models.csv
    :header-rows: 1

.. |c| unicode:: U+2705
.. |x| unicode:: U+274C
