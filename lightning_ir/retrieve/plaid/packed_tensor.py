from pathlib import Path
from typing import Sequence, Tuple

import torch


class PackedTensor:
    def __init__(self, packed_tensor: torch.Tensor, lengths: Sequence[int]) -> None:
        self.packed_tensor = packed_tensor
        self.lengths = list(lengths)
        self._segmented_tensor: Tuple[torch.Tensor, ...] | None = None

    def __repr__(self) -> str:
        return f"PackedTensor(packed_tensor={self.packed_tensor}, lengths={self.lengths})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def segmented_tensor(self) -> Tuple[torch.Tensor, ...]:
        if self._segmented_tensor is None:
            self._segmented_tensor = torch.split(self.packed_tensor, self.lengths)
        return self._segmented_tensor

    def lookup(
        self, packed_idcs: torch.Tensor, idcs_lengths: Sequence[int] | int, unique: bool = False
    ) -> "PackedTensor":
        output_tensors = []
        lengths = []
        for lookup_idcs in torch.split(packed_idcs, idcs_lengths):
            intermediate_tensors = []
            for idx in lookup_idcs:
                intermediate_tensors.append(self.segmented_tensor[idx])

            cat_tensors = torch.cat(intermediate_tensors)
            if unique:
                cat_tensors = torch.unique(cat_tensors)
            lengths.append(cat_tensors.shape[0])
            output_tensors.append(cat_tensors)

        return PackedTensor(torch.cat(output_tensors), lengths)

    def to_padded_tensor(self, pad_value: int = 0) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(self.segmented_tensor, batch_first=True, padding_value=pad_value)
