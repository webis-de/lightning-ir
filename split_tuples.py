from argparse import ArgumentParser
from pathlib import Path

import ir_datasets
import pandas as pd
from ir_datasets.formats import GenericDocPair
from tqdm import tqdm

from mvr.data import ScoredDocTuple, register_colbert_docpairs, register_kd_docpairs

register_kd_docpairs()
register_colbert_docpairs()


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", type=int, required=True)

    args = parser.parse_args(args)

    dataset = ir_datasets.load(args.dataset)
    val_query_ids = set()

    train_file = args.output_dir.joinpath(
        "__train__" + args.dataset.replace("/", "-") + ".tsv"
    )
    val_file = args.output_dir.joinpath(
        "__val__" + args.dataset.replace("/", "-") + ".tsv"
    )
    with train_file.open("w") as train_f:
        with val_file.open("w") as val_f:
            for sample in tqdm(dataset.docpairs_iter(), total=dataset.docpairs_count()):
                query_id = sample.query_id
                if isinstance(sample, ScoredDocTuple):
                    doc_ids = sample.doc_ids
                    scores = sample.scores
                elif isinstance(sample, GenericDocPair):
                    doc_ids = (sample.doc_id_a, sample.doc_id_b)
                    scores = None
                if len(val_query_ids) < args.split:
                    val_query_ids.add(query_id)
                if query_id in val_query_ids:
                    f = val_f
                else:
                    f = train_f
                scores_str = (
                    "\t".join(str(score) for score in scores)
                    if scores is not None
                    else ""
                )
                data = f"{query_id}\t" + "\t".join(doc_ids) + "\t" + scores_str + "\n"
                f.write(data)


if __name__ == "__main__":
    main()
