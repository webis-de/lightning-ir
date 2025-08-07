.. _model-zoo:

=========
Model Zoo
=========

The following table lists models from the `HuggingFace Model Hub <https://huggingface.co/models>`_ that are supported in Lightning IR.

Native models were fine-tuned using Lightning IR and the model's HuggingFace model card provides Lightning IR configurations for reproduction. Non-native models were fine-tuned externally but are supported in Lightning IR for inference.



**Reranking Results**

For each model, the table reports the re-ranking effectiveness in terms of nDCG\@10 on the officially released run files containing 1,000 passages for TREC Deep Learning 2019 and 2020. 

``Reproduction``

The following command can be used to reproduce the results:

.. code-block:: bash

    lightning-ir re_rank --config config.yaml

.. collapse:: config.yaml
    
    .. code-block:: yaml

        trainer:
          logger: false
          enable_checkpointing: false
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

.. csv-table::
  :file: ./models/reranking.csv
  :header-rows: 1

**Retrieval Results**

For each model, the table reports the retrieval effectiveness in terms of nDCG\@10 on the NanoBEIR dataset. The scores are reported on the 13 individual datasets of NanoBEIR and averaged to get an overall score for each model.

``Reproduction``

The following command can be used to reproduce the results:

.. code-block:: bash

    lightning-ir index --config index.yaml # for indexing
    lightning-ir search --config search.yaml # for retrieval

.. collapse:: index.yaml
    
    .. code-block:: yaml

        trainer:
          accelerator: auto
          strategy: auto
          devices: auto
          num_nodes: 1
          logger: false
          callbacks:
            - class_path: IndexCallback
              init_args:
                index_dir: {INDEX_DIR}
                index_config:
                  class_path: TorchDenseIndexConfig # for Bi-Encoder and ColBERT
                  # class_path: TorchSparseIndexConfig # for SPLADE
            
        model:
          class_path: BiEncoderModule
          init_args:
            model_name_or_path: {MODEL_NAME}
            evaluation_metrics:
            - nDCG@10
        data:
          class_path: LightningIRDataModule
          init_args:
            num_workers: 1
            inference_batch_size: 128
            inference_datasets:
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/climate-fever
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/dbpedia-entity
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/fever
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/fiqa
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/hotpotqa
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/msmarco
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/nfcorpus
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/nq
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/quora
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/scidocs
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/arguana
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/scifact
            - class_path: DocDataset
              init_args:
                doc_dataset: nano-beir/webis-touche2020

.. collapse:: search.yaml

    .. code-block:: yaml

        trainer:
          accelerator: auto
          strategy: auto
          devices: auto
          num_nodes: 1
          logger: false
          callbacks:
            - class_path: SearchCallback
              init_args:
                search_config:
                  class_path: TorchDenseSearchConfig # for Bi-Encoder and ColBERT
                  # class_path: TorchSparseSearchConfig # for SPLADE
                  init_args:
                    k: 100
                index_dir: {INDEX_DIR}
                use_gpu: true
                save_dir: ./runs
        model:
          class_path: BiEncoderModule
          init_args:
            model_name_or_path: {MODEL_NAME}
            evaluation_metrics:
            - nDCG@10
        data:
          class_path: LightningIRDataModule
          init_args:
            num_workers: 1
            inference_batch_size: 8
            inference_datasets:
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/climate-fever
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/dbpedia-entity
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/fever
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/fiqa
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/hotpotqa
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/msmarco
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/nfcorpus
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/nq
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/quora
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/scidocs
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/arguana
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/scifact
            - class_path: QueryDataset
              init_args:
                query_dataset: nano-beir/webis-touche2020

.. csv-table::
  :file: ./models/retrieval.csv
  :header-rows: 1

.. |c| unicode:: U+2705
.. |x| unicode:: U+274C
