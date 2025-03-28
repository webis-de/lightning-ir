import numpy as np
import torch


class SeismicFormatConverter:

    @staticmethod
    def convert_to_seismic_format(embeddings: torch.Tensor) -> bytes:
        if embeddings.ndim != 2:
            raise ValueError("Expected 2D tensor")
        batch_idcs, term_idcs = embeddings.nonzero(as_tuple=True)
        lengths = torch.bincount(batch_idcs).tolist()
        values = embeddings[(batch_idcs, term_idcs)]

        out = b""
        for t, v in zip(term_idcs.split(lengths), values.split(lengths)):
            out += (len(t)).to_bytes(4, byteorder="little", signed=False)
            out += t.numpy(force=True).astype(np.int32).tobytes()
            out += v.numpy(force=True).astype(np.float32).tobytes()
        return out
