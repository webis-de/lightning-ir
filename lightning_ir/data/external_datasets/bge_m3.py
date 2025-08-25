import codecs
import json

from lightning_ir.data.external_datasets.ir_datasets_utils import ScoredDocTuples, register_new_dataset


class BGEM3ScoredDocTuples(ScoredDocTuples):

    def docpairs_iter(self):
        with self._docpairs_dlc.stream() as f:
            f = codecs.getreader("utf8")(f)
            for line in f:
                data = json.loads(line)
                # TODO parse data and yield the docpairs


def register_bge_m3_docpairs():
    dlc_contents = {
        "url": "https://huggingface.co/datasets/Shitao/bge-m3-data/resolve/main/bge-m3-data.tar.gz?download=true",
        "expected_md5": "TODO",
        "cache_path": "msmarco-passage/train/sbert-docpairs.jsonl.gz",
    }
    # TODO the data is split into multiple jsonl files gzipped, need to handle that
    # write into new single file and shuffle the lines?! also the docs needs to be extracted, see how MS~MARCO does this
    register_new_dataset(
        "msmarco-passage/train/sbert-docpairs",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        docpairs=dlc_contents,
        DocpairsType=BGEM3ScoredDocTuples,
    )


register_bge_m3_docpairs()


if __name__ == "__main__":
    import ir_datasets

    dataset = ir_datasets.load("bge-m3")


sbert_dataset = SBERTScordDocTuples("msmarco-passage/train/sbert-docpairs")
print(next(iter(sbert_dataset)))
