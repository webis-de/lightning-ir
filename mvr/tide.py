from mvr.mvr import MVRModel, MVRConfig


class TIDEConfig(MVRConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
