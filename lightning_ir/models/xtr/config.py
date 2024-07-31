from typing import Literal

from ..col import ColConfig


class XTRConfig(ColConfig):
    model_type = "xtr"

    ADDED_ARGS = ColConfig.ADDED_ARGS.union({"token_retrieval_k", "fill_strategy", "normalization"})

    def __init__(
        self,
        token_retrieval_k: int | None = None,
        fill_strategy: Literal["zero", "min"] = "zero",
        normalization: Literal["Z"] | None = "Z",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.token_retrieval_k = token_retrieval_k
        self.fill_strategy = fill_strategy
        self.normalization = normalization
