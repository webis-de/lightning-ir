from tokenizers.processors import TemplateProcessing

class MVRTokenizer(BiEncoderTokenizer):
    def __init__(
            self,
            num_viewer_tokens: int = 8,
    ):
        super().__init__()
        if num_viewer_tokens is not None:
            viewer_tokens = " ".join(f"[VIE{idx}]" for idx in range(num_viewer_tokens))
            self.add_tokens(viewer_tokens, special_tokens=True)
            viewer_token_ids = [(f"[VIE{viewer_token_id}]", self.viewer_token_id(f"[VIE{viewer_token_id}]")) for viewer_token_id in range(num_viewer_tokens)]
            self.doc_post_processor = TemplateProcessing(
                single=f"{viewer_tokens} $0 [SEP]",
                pair=f"[CLS] {self.QUERY_TOKEN} $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    *viewer_token_ids,
                ],
            )

    
    def viewer_token_id(self, viewer_token_id) -> int | None:
        """The token id of the query token if marker tokens are added.

        :return: Token id of the query token
        :rtype: int | None
        """
        if f"[VIE{viewer_token_id}]" in self.added_tokens_encoder:
            return self.added_tokens_encoder[f"[VIE{viewer_token_id}]"]
        return None