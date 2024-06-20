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
    MultiVectorBiEncoderConfig,
    MultiVectorBiEncoderModel,
    ScoringFunction,
    SingleVectorBiEncoderConfig,
    SingleVectorBiEncoderModel,
)
from .cross_encoder import (
    CrossEncoderConfig,
    CrossEncoderModel,
    CrossEncoderModule,
    CrossEncoderOutput,
    CrossEncoderTokenizer,
)
from .data import (
    BiEncoderRunBatch,
    CrossEncoderRunBatch,
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
AutoConfig.register(SingleVectorBiEncoderConfig.model_type, SingleVectorBiEncoderConfig)
AutoModel.register(SingleVectorBiEncoderConfig, SingleVectorBiEncoderModel)
AutoConfig.register(MultiVectorBiEncoderConfig.model_type, MultiVectorBiEncoderConfig)
AutoModel.register(MultiVectorBiEncoderConfig, MultiVectorBiEncoderModel)

AutoConfig.register(CrossEncoderConfig.model_type, CrossEncoderConfig)
AutoModel.register(CrossEncoderConfig, CrossEncoderModel)


__all__ = [
    "ApproxMRR",
    "ApproxNDCG",
    "ApproxRankMSE",
    "BiEncoderConfig",
    "BiEncoderModel",
    "BiEncoderModule",
    "BiEncoderOutput",
    "BiEncoderRunBatch",
    "BiEncoderTokenizer",
    "ConstantMarginMSE",
    "ConstantSchedulerWithWarmup",
    "CrossEncoderConfig",
    "CrossEncoderModel",
    "CrossEncoderModule",
    "CrossEncoderOutput",
    "CrossEncoderRunBatch",
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
    "MultiVectorBiEncoderConfig",
    "MultiVectorBiEncoderModel",
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
    "SingleVectorBiEncoderConfig",
    "SingleVectorBiEncoderModel",
    "SupervisedMarginMSE",
    "TupleDataset",
    "WarmupScheduler",
]
