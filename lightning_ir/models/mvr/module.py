from typing import Sequence, Tuple
from lightning_ir.data.data import IndexBatch, RankBatch, SearchBatch
from lightning_ir.loss.loss import LossFunction
from ...bi_encoder import BiEncoderModule
from lightning_ir.models.mvr.config import MVRConfig
from lightning_ir.models.mvr.model import MVRModel, MVROutput


class MVRModule(BiEncoderModule):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: MVRConfig | None = None,
        model: MVRModel | None = None,
        loss_functions: Sequence[LossFunction | Tuple[LossFunction, float]] | None = None,
    ):
        super().__init__(model_name_or_path=model_name_or_path, config=config, model=model, loss_functions=loss_functions, evaluation_metrics=None, index_dir=None, search_config=None)
        if config.num_viewer_tokens and len(self.tokenizer) > self.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), 8)


    def forward(self, batch: RankBatch | IndexBatch | SearchBatch) -> MVROutput:

        queries = getattr(batch, "queries", None)
        docs = getattr(batch, "docs", None)
        num_docs = None
        if isinstance(batch, RankBatch):
            num_docs = None if docs is None else [len(d) for d in docs]
            docs = [d for nested in docs for d in nested] if docs is not None else None
        encodings = self.prepare_input(queries, docs, num_docs)

        if not encodings:
            raise ValueError("No encodings were generated.")
        output = self.model.forward(
            encodings.get("query_encoding", None), encodings.get("doc_encoding", None), num_docs
        )
        if isinstance(batch, SearchBatch) and self.searcher is not None:
            scores, doc_ids, num_docs = self.searcher.search(output)
            output.scores = scores
            cum_num_docs = [0] + [sum(num_docs[: i + 1]) for i in range(len(num_docs))]
            doc_ids = tuple(tuple(doc_ids[cum_num_docs[i] : cum_num_docs[i + 1]]) for i in range(len(num_docs)))
            batch.doc_ids = doc_ids
        return output