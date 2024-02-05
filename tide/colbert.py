from pathlib import Path

from transformers import AutoModel

from .loss import LossFunction
from .mvr import MVRConfig, MVRModel, MVRModule


class ColBERTModel(MVRModel):
    def __init__(
        self, model_name_or_path: Path | str, config: MVRConfig | None
    ) -> None:
        model = AutoModel.from_pretrained(model_name_or_path)
        if config is not None:
            model.config.update(config.to_diff_dict())
        super().__init__(model)


class ColBERTModule(MVRModule):
    def __init__(
        self,
        model: ColBERTModel,
        loss_function: LossFunction,
    ) -> None:
        super().__init__(model, loss_function)
