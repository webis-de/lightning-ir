# Lightning IR

Your one-stop shop for fine-tuning and running neural ranking models.

-----------------

Lightning IR is a library for fine-tuning and running neural ranking models. It is built on top of [Lightning](https://lightning.ai/docs/pytorch/stable/) to provide a simple and flexible interface to interact with neural ranking models.

Two types of models are supported: cross-encoders and bi-encoders. Cross-encoders are models that encode a query and a document together (monoBERT, monoT5, RankT5, etc.), while bi-encoders encode queries and documents separately (DPR, ColBERT, SPLADE, etc.).

Both types of models are usually trained on the same types of data: triples of queries, positive documents, and negative documents. Therefore, the library provides a unified interface to fine-tune and run both types of models. See the [Fine-tuning](#fine-tuning) section for more details.

Regarding inference, since bi-encoders encode queries and documents separately, they can be used to index documents and search for relevant documents. Lightning IR provides a simple interface for indexing and searching with bi-encoders. See the [Indexing](#indexing) and [Searching](#searching) sections for more details. Cross-encoders, on the other hand, encode queries and documents together, making them only suitable for re-ranking. Lightning IR provides a simple interface for re-ranking with cross-encoders and bi-encoders. See the [Re-ranking](#re-ranking) section for more details.

## Installation

We're currently in the process of setting up the package on PyPI. In the meantime, you can install the package from source.

```bash
git clone
cd lightning-ir
pip install .
```

## Model Zoo

### Cross-encoders

| Model Name                                                          | TREC DL 19/20 nDCG@10 (BM25) | TIREx nDCG@10 |
| ------------------------------------------------------------------- | ---------------------------- | ------------- |
| [monoelectra-base](https://huggingface.co/webis/monoelectra-base)   | 0.715                        | 0.416         |
| [monoelectra-large](https://huggingface.co/webis/monoelectra-large) | 0.730                        | 0.434         |
| monoT5 (Coming soon)                                                | --                           | --            |

### Bi-encoders

| Model Name            | TREC DL 19/20 nDCG@10 |
| --------------------- | --------------------- |
| ColBERT (Coming soon) | --                    |
| DPR (Coming soon)     | --                    |
| SPLADE (Coming soon)  | --                    |

## Usage

### Command Line Interface

Lightning IR uses the [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli) and adds some additional options to provide a unified interface for fine-tuning and running neural ranking models. After installation, the CLI can be accessed via the `lightning-ir` command.

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

### Configuration

### Data Formats

### Examples

#### Fine-tuning

#### Indexing

#### Searching

#### Re-ranking
