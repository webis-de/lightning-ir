from transformers import AutoConfig, AutoModel

from .base import (
    LightningIRConfig,
    LightningIRModel,
    LightningIRModelClassFactory,
    LightningIRModule,
    LightningIROutput,
    LightningIRTokenizer,
)
from .bi_encoder import (
    BiEncoderConfig,
    BiEncoderModel,
    BiEncoderModule,
    BiEncoderOutput,
    BiEncoderTokenizer,
    ScoringFunction,
)
from .cross_encoder import (
    CrossEncoderConfig,
    CrossEncoderModel,
    CrossEncoderModule,
    CrossEncoderOutput,
    CrossEncoderTokenizer,
)
from .data import (
    DocDataset,
    DocSample,
    IndexBatch,
    LightningIRDataModule,
    QueryDataset,
    QuerySample,
    RunDataset,
    RunSample,
    SearchBatch,
    TrainBatch,
    TupleDataset,
)
from .lightning_utils import (
    LR_SCHEDULERS,
    ConstantSchedulerWithWarmup,
    IndexCallback,
    LinearSchedulerWithWarmup,
    ReRankCallback,
    SearchCallback,
    WarmupScheduler,
)
from .loss import (
    ApproxMRR,
    ApproxNDCG,
    ApproxRankMSE,
    ConstantMarginMSE,
    FLOPSRegularization,
    InBatchCrossEntropy,
    KLDivergence,
    L1Regularization,
    L2Regularization,
    LocalizedContrastiveEstimation,
    RankNet,
    SupervisedMarginMSE,
)
from .models import ColConfig, ColModel, SpladeConfig, SpladeModel
from .retrieve import (
    FaissFlatIndexConfig,
    FaissFlatIndexer,
    FaissIVFPQIndexConfig,
    FaissIVFPQIndexer,
    FaissSearchConfig,
    Indexer,
    SearchConfig,
    Searcher,
    SparseIndexConfig,
    SparseIndexer,
    SparseSearchConfig,
)

AutoConfig.register(BiEncoderConfig.model_type, BiEncoderConfig)
AutoModel.register(BiEncoderConfig, BiEncoderModel)
AutoConfig.register(CrossEncoderConfig.model_type, CrossEncoderConfig)
AutoModel.register(CrossEncoderConfig, CrossEncoderModel)
AutoConfig.register(ColConfig.model_type, ColConfig)
AutoModel.register(ColConfig, ColModel)
AutoConfig.register(SpladeConfig.model_type, SpladeConfig)
AutoModel.register(SpladeConfig, SpladeModel)


__all__ = [
    "ApproxMRR",
    "ApproxNDCG",
    "ApproxRankMSE",
    "BiEncoderConfig",
    "BiEncoderModel",
    "BiEncoderModule",
    "BiEncoderOutput",
    "BiEncoderTokenizer",
    "ColConfig",
    "ColModel",
    "ConstantMarginMSE",
    "ConstantSchedulerWithWarmup",
    "CrossEncoderConfig",
    "CrossEncoderModel",
    "CrossEncoderModule",
    "CrossEncoderOutput",
    "CrossEncoderTokenizer",
    "DocDataset",
    "DocSample",
    "FaissFlatIndexConfig",
    "FaissFlatIndexer",
    "FaissIVFPQIndexConfig",
    "FaissIVFPQIndexer",
    "FaissSearchConfig",
    "FLOPSRegularization",
    "InBatchCrossEntropy",
    "IndexBatch",
    "IndexCallback",
    "Indexer",
    "KLDivergence",
    "L1Regularization",
    "L2Regularization",
    "LightningIRConfig",
    "LightningIRDataModule",
    "LightningIRModel",
    "LightningIRModelClassFactory",
    "LightningIRModule",
    "LightningIROutput",
    "LightningIRTokenizer",
    "LinearSchedulerWithWarmup",
    "LocalizedContrastiveEstimation",
    "LR_SCHEDULERS",
    "QueryDataset",
    "QuerySample",
    "RankNet",
    "ReRankCallback",
    "RunDataset",
    "RunSample",
    "ScoringFunction",
    "SearchBatch",
    "SearchCallback",
    "SearchConfig",
    "Searcher",
    "SparseIndexConfig",
    "SparseIndexer",
    "SparseSearchConfig",
    "SupervisedMarginMSE",
    "TrainBatch",
    "TupleDataset",
    "WarmupScheduler",
]
