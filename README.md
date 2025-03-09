# Lightning IR

<p align="center">
<img src="./docs/_static/lightning-ir-logo.svg" alt="lightning ir logo" width="10%">
<p align="center">Your one-stop shop for fine-tuning and running neural ranking models.</p>
</p>

-----------------

Lightning IR is a library for fine-tuning and running neural ranking models. It is built on top of [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to provide a simple and flexible interface to interact with neural ranking models.

Want to:

- fine-tune your own cross- or bi-encoder models?
- index and search through a collection of documents with ColBERT or SPLADE?
- re-rank documents with state-of-the-art models?

Lightning IR has you covered!
  
## Installation

Lightning IR can be installed using pip:

```
pip install lightning-ir
```

## Getting Started

See the [Quickstart](https://webis-de.github.io/lightning-ir/quickstart.html) guide for an introduction to Lightning IR. The [Documentation](https://webis-de.github.io/lightning-ir/) provides a detailed overview of the library's functionality.

The easiest way to use Lightning IR is via the CLI. It uses the [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli) and adds additional options to provide a unified interface for fine-tuning and running neural ranking models.

The behavior of the CLI can be customized using yaml configuration files. See the [configs](configs) directory for several example configuration files. For example, the following command can be used to re-rank the official TREC DL 19/20 re-ranking set with a pre-finetuned cross-encoder model. It will automatically download the model and data, run the re-ranking, write the results to a TREC-style run file, and report the nDCG@10 score.

```bash
lightning-ir re_rank \
  --config ./configs/trainer/inference.yaml \
  --config ./configs/callbacks/rank.yaml \
  --config ./configs/data/re-rank-trec-dl.yaml \
  --config ./configs/models/monoelectra.yaml
```

For more details, see the [Usage](#usage) section.

## Usage

### Command Line Interface

The CLI offers four subcommands:

```
$ lightning-ir -h
Lightning Trainer command line tool

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    fit                 Runs the full optimization routine.
    index               Index a collection of documents.
    search              Search for relevant documents.
    re_rank             Re-rank a set of retrieved documents.
```

Configurations files need to be provided to specify model, data, and fine-tuning/inference parameters. See the [configs](configs) directory for examples. Four types of configurations exists:

- `trainer`: Specifies the fine-tuning/inference parameters and callbacks.
- `model`: Specifies the model to use and its parameters.
- `data`: Specifies the dataset(s) to use and its parameters.
- `optimizer`: Specifies the optimizer parameters (only needed for fine-tuning).

### Example

The following example demonstrates how to fine-tune a BERT-based single-vector bi-encoder model using the official MS MARCO triples. The fine-tuned model is then used to index the MS MARCO passage collection and search for relevant passages. Finally, we show how to re-rank the retrieved passages.

#### Fine-tuning

To fine-tune a bi-encoder model on the MS MARCO triples dataset, use the following configuration file and command:

<details>

<summary>bi-encoder-fit.yaml</summary>

```yaml
trainer:
  callbacks:
  - class_path: ModelCheckpoint
  max_epochs: 1
  max_steps: 100000
data:
  class_path: LightningIRDataModule
  init_args:
    train_batch_size: 32
    train_dataset:
      class_path: TupleDataset
      init_args:
        tuples_dataset: msmarco-passage/train/triples-small
model:
  class_path: BiEncoderModule
  init_args:
    model_name_or_path: bert-base-uncased
    config:
      class_path: BiEncoderConfig
    loss_functions:
    - class_path: RankNet
optimizer:
  class_path: AdamW
  init_args:
    lr: 1e-5
```

</details>

```bash
lightning-ir fit --config bi-encoder-fit.yaml
```

The fine-tuned model is saved in the directory `lightning_logs/version_X/huggingface_checkpoint/`.

#### Indexing

We now assume the model from the previous fine-tuning step was moved to the directory `models/bi-encoder`. To index the MS MARCO passage collection with [faiss](https://github.com/facebookresearch/faiss) using the fine-tuned model, use the following configuration file and command:

<details>

<summary>bi-encoder-index.yaml</summary>

```yaml
trainer:
  callbacks:
  - class_path: IndexCallback
    init_args:
        index_config:
          class_path: FaissFlatIndexConfig
model:
  class_path: BiEncoderModule
  init_args:
    model_name_or_path: models/bi-encoder
data:
  class_path: LightningIRDataModule
  init_args:
    num_workers: 1
    inference_batch_size: 256
    inference_datasets:
    - class_path: DocDataset
      init_args:
        doc_dataset: msmarco-passage
```

</details>

```bash
lightning-ir index --config bi-encoder-index.yaml
```

The index is saved in the directory `models/bi-encoder/indexes/msmarco-passage`.

#### Searching

To search for relevant documents in the MS MARCO passage collection using the bi-encoder and index, use the following configuration file and command:

<details>

<summary>bi-encoder-search.yaml</summary>

```yaml
trainer:
  callbacks:
  - class_path: RankCallback
model:
  class_path: BiEncoderModule
  init_args:
    model_name_or_path: models/bi-encoder
    index_dir: models/bi-encoder/indexes/msmarco-passage
    search_config:
      class_path: FaissFlatSearchConfig
      init_args:
        k: 100
    evaluation_metrics:
    - nDCG@10
data:
  class_path: LightningIRDataModule
  init_args:
    num_workers: 1
    inference_batch_size: 4
    inference_datasets:
    - class_path: QueryDataset
      init_args:
        query_dataset: msmarco-passage/trec-dl-2019/judged
    - class_path: QueryDataset
      init_args:
        query_dataset: msmarco-passage/trec-dl-2020/judged
```

</details>

```bash
lightning-ir search --config bi-encoder-search.yaml
```

The run files are saved as `models/bi-encoder/runs/msmarco-passage-trec-dl-20XX.run`. Additionally, the nDCG@10 scores are printed to the console.

#### Re-ranking

Assuming we've also fine-tuned a cross-encoder that is saved in the directory `models/cross-encoder`, we can re-rank the retrieved documents using the following configuration file and command:

<details>

<summary>cross-encoder-re-rank.yaml</summary>

```yaml
trainer:
  callbacks:
  - class_path: RankCallback
model:
  class_path: CrossEncoderModule
  init_args:
    model_name_or_path: models/cross-encoder
    evaluation_metrics:
    - nDCG@10
data:
  class_path: LightningIRDataModule
  init_args:
    num_workers: 1
    inference_batch_size: 4
    inference_datasets:
    - class_path: RunDataset
      init_args:
        run_path_or_id: models/bi-encoder/runs/msmarco-passage-trec-dl-2019.run
        depth: 100
        sample_size: 100
        sampling_strategy: top
    - class_path: RunDataset
      init_args:
        run_path_or_id: models/bi-encoder/runs/msmarco-passage-trec-dl-2020.run
        depth: 100
        sample_size: 100
        sampling_strategy: top
```

</details>

```bash
lightning-ir re_rank --config cross-encoder-re-rank.yaml
```

The run files are saved as `models/cross-encoder/runs/msmarco-passage-trec-dl-20XX.run`. Additionally, the nDCG@10 scores are printed to the console.
