from ...bi_encoder import BiEncoderModule
from lightning_ir.models.mvr.config import MVRConfig


class MVRModule(BiEncoderModule):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: MVRConfig | None = None,
    ):
        super().__init__(model_name_or_path=model_name_or_path, config=config, model=None, loss_functions=None, evaluation_metrics=None, index_dir=None, search_config=None)
        if config.num_viewer_tokens and len(self.tokenizer) > self.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), 8)
