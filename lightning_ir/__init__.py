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
    TrainBatch,
    DocDataset,
    DocSample,
    IndexBatch,
    LightningIRDataModule,
    QueryDataset,
    QuerySample,
    RunDataset,
    RunSample,
    SearchBatch,
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
    InBatchCrossEntropy,
    KLDivergence,
    LocalizedContrastiveEstimation,
    RankNet,
    SupervisedMarginMSE,
)
from .models import ColConfig, ColModel
from .retrieve import (
    FlatIndexConfig,
    FlatIndexer,
    Indexer,
    IVFPQIndexConfig,
    IVFPQIndexer,
    SearchConfig,
    Searcher,
)

AutoConfig.register(BiEncoderConfig.model_type, BiEncoderConfig)
AutoModel.register(BiEncoderConfig, BiEncoderModel)
AutoConfig.register(CrossEncoderConfig.model_type, CrossEncoderConfig)
AutoModel.register(CrossEncoderConfig, CrossEncoderModel)
AutoConfig.register(ColConfig.model_type, ColConfig)
AutoModel.register(ColConfig, ColModel)


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
    "FlatIndexConfig",
    "FlatIndexer",
    "InBatchCrossEntropy",
    "IndexBatch",
    "IndexCallback",
    "Indexer",
    "IVFPQIndexConfig",
    "IVFPQIndexer",
    "KLDivergence",
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
    "SupervisedMarginMSE",
    "TrainBatch",
    "TupleDataset",
    "WarmupScheduler",
]
