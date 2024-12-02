from tokenizers.processors import TemplateProcessing

class MVRTokenizer(BiEncoderTokenizer):
    def __init__(
            self,
            num_viewer_tokens: int = 8,
    ):
        super().__init__()
        if num_viewer_tokens is not None:
            viewer_tokens = " ".join(f"[VIE{idx}]" for idx in num_viewer_tokens)
            self.add_tokens(viewer_tokens, special_tokens=True)
            self.doc_post_processor = TemplateProcessing(
                single=f"{viewer_tokens} {self.DOC_TOKEN} $0 [SEP]",
                pair=f"[CLS] {self.QUERY_TOKEN} $A [SEP] {self.DOC_TOKEN} $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    (self.QUERY_TOKEN, self.query_token_id),
                    (self.DOC_TOKEN, self.doc_token_id),
                ],
            )