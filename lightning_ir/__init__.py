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
    RankSample,
    RunDataset,
    SearchBatch,
    TrainBatch,
    TupleDataset,
)
from .lightning_utils import (
    LR_SCHEDULERS,
    ConstantLRSchedulerWithLinearWarmup,
    GenericConstantSchedulerWithLinearWarmup,
    GenericConstantSchedulerWithQuadraticWarmup,
    GenericLinearSchedulerWithLinearWarmup,
    IndexCallback,
    LinearLRSchedulerWithLinearWarmup,
    RankCallback,
    WarmupLRScheduler,
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
from .models import ColConfig, ColModel, SpladeConfig, SpladeModel, XTRConfig, XTRModel
from .retrieve import (
    FaissFlatIndexConfig,
    FaissFlatIndexer,
    FaissIVFIndexConfig,
    FaissIVFIndexer,
    FaissIVFPQIndexConfig,
    FaissIVFPQIndexer,
    FaissSearchConfig,
    FaissSearcher,
    IndexConfig,
    Indexer,
    SearchConfig,
    Searcher,
    SparseIndexConfig,
    SparseIndexer,
    SparseSearchConfig,
    SparseSearcher,
)

AutoConfig.register(BiEncoderConfig.model_type, BiEncoderConfig)
AutoModel.register(BiEncoderConfig, BiEncoderModel)
AutoConfig.register(CrossEncoderConfig.model_type, CrossEncoderConfig)
AutoModel.register(CrossEncoderConfig, CrossEncoderModel)
AutoConfig.register(ColConfig.model_type, ColConfig)
AutoModel.register(ColConfig, ColModel)
AutoConfig.register(SpladeConfig.model_type, SpladeConfig)
AutoModel.register(SpladeConfig, SpladeModel)
AutoConfig.register(XTRConfig.model_type, XTRConfig)
AutoModel.register(XTRConfig, XTRModel)

__version__ = "0.0.1"

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
    "ConstantLRSchedulerWithLinearWarmup",
    "ConstantMarginMSE",
    "CrossEncoderConfig",
    "CrossEncoderModel",
    "CrossEncoderModule",
    "CrossEncoderOutput",
    "CrossEncoderTokenizer",
    "DocDataset",
    "DocSample",
    "FaissFlatIndexConfig",
    "FaissFlatIndexer",
    "FaissIVFIndexConfig",
    "FaissIVFIndexer",
    "FaissIVFPQIndexConfig",
    "FaissIVFPQIndexer",
    "FaissSearchConfig",
    "FaissSearcher",
    "FLOPSRegularization",
    "GenericConstantSchedulerWithLinearWarmup",
    "GenericConstantSchedulerWithQuadraticWarmup",
    "GenericLinearSchedulerWithLinearWarmup",
    "InBatchCrossEntropy",
    "IndexBatch",
    "IndexCallback",
    "IndexConfig",
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
    "LinearLRSchedulerWithLinearWarmup",
    "LocalizedContrastiveEstimation",
    "LR_SCHEDULERS",
    "QueryDataset",
    "QuerySample",
    "RankCallback",
    "RankNet",
    "RunDataset",
    "RankSample",
    "ScoringFunction",
    "SearchBatch",
    "SearchConfig",
    "Searcher",
    "SparseIndexConfig",
    "SparseIndexer",
    "SparseSearchConfig",
    "SparseSearcher",
    "SupervisedMarginMSE",
    "TrainBatch",
    "TupleDataset",
    "WarmupLRScheduler",
    "XTRConfig",
    "XTRModel",
]
