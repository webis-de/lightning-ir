if simulate_token_retrieval and self.xtr_token_retrieval_k is not None:
    # TODO fix for new reduction
    # TODO move to xtr
    ib_similarity = similarity.transpose(1, 2).reshape(batch_size, query_len, -1)
    top_k_similarity = ib_similarity.topk(self.xtr_token_retrieval_k, dim=-1)
    cut_off_similarity = top_k_similarity.values[..., -1]
    similarity = similarity.masked_fill(
        similarity < cut_off_similarity[:, None, :, None], 0
    )
