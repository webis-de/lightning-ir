class MVRTokenizer(BiEncoderTokenizer):
    def __init__(
            self,
            add_viewer_tokens: bool = True,
    ):
        super().__init__()
        