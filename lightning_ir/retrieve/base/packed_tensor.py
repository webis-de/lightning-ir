from typing import Sequence, Tuple

import torch


class PackedTensor(torch.Tensor):

    def __new__(cls, *args, lengths: Sequence[int] | None = None, **kwargs) -> "PackedTensor":
        if lengths is None:
            raise ValueError("lengths must be provided")
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, lengths: Sequence[int] | None = None, **kwargs) -> None:
        if lengths is None:
            raise ValueError("lengths must be provided")
        if sum(lengths) != len(self):
            raise ValueError("Sum of lengths must equal the length of the tensor")
        self.lengths = list(lengths)
        self._segmented_tensor: Tuple[torch.Tensor, ...] | None = None

    @property
    def segmented_tensor(self) -> Tuple[torch.Tensor, ...]:
        if self._segmented_tensor is None:
            self._segmented_tensor = torch.split(self, self.lengths)
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

        return PackedTensor(torch.cat(output_tensors), lengths=lengths)

    def to_padded_tensor(self, pad_value: int = 0) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(self.segmented_tensor, batch_first=True, padding_value=pad_value)
