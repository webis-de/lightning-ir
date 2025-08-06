from .colbert import register_colbert_docpairs
from .cross_architecture_knowledge_distillation import register_kd_docpairs
from .ir_datasets_utils import register_new_dataset
from .rank_distillm import register_rank_distillm


def _register_external_datasets():
    register_kd_docpairs()
    register_colbert_docpairs()
    register_rank_distillm()


__all__ = ["register_new_dataset"]
