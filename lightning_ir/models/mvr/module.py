class MVRModule(BiEncoderModule):
    def __init__(
            self,
            
    ):
        super().__init__()
        if self.config.add_viewer_tokens and len(self.tokenizer) > self.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), 8)