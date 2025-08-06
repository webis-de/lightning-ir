"""SeismicFormatConverter class for converting embeddings to a seismic format."""

import numpy as np
import torch


class SeismicFormatConverter:
    """Converter for embeddings to a seismic format."""

    @staticmethod
    def convert_to_seismic_format(embeddings: torch.Tensor) -> bytes:
        """Convert embeddings to a seismic format.

        Args:
            embeddings (torch.Tensor): The embeddings to convert.
        Returns:
            bytes: The converted embeddings in seismic format.
        Raises:
            ValueError: If the embeddings are not 2D.
        """
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
