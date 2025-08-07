"""PackedTensor class for handling tensors with variable segment lengths."""

from typing import Sequence, Tuple

import torch


class PackedTensor(torch.Tensor):
    """A tensor that contains a sequence of tensors with varying lengths."""

    def __new__(cls, *args, lengths: Sequence[int] | None = None, **kwargs) -> "PackedTensor":
        """Create a new PackedTensor instance.

        Args:
            lengths (Sequence[int] | None): A sequence of lengths for each segment in the tensor. If provided, the
                tensor must be created with a total length equal to the sum of these lengths. Defaults to None.
        Returns:
            PackedTensor: A new instance of PackedTensor.
        Raises:
            ValueError: If lengths is None.
        """
        if lengths is None:
            raise ValueError("lengths must be provided")
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, lengths: Sequence[int] | None = None, **kwargs) -> None:
        """Initialize the PackedTensor instance.

        Args:
            lengths (Sequence[int] | None): A sequence of lengths for each segment in the tensor. If provided, the
                tensor must be created with a total length equal to the sum of these lengths. Defaults to None.
        Raises:
            ValueError: If lengths is None.
            ValueError: If the sum of lengths does not equal the length of the tensor.
        """
        if lengths is None:
            raise ValueError("lengths must be provided")
        if sum(lengths) != len(self):
            raise ValueError("Sum of lengths must equal the length of the tensor")
        self.lengths = list(lengths)
        self._segmented_tensor: Tuple[torch.Tensor, ...] | None = None

    @property
    def segmented_tensor(self) -> Tuple[torch.Tensor, ...]:
        """Get the segmented tensor, which is a tuple of tensors split according to the specified lengths.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple of tensors, each corresponding to a segment defined by the lengths.
        """
        if self._segmented_tensor is None:
            self._segmented_tensor = torch.split(self, self.lengths)
        return self._segmented_tensor

    def lookup(
        self, packed_idcs: torch.Tensor, idcs_lengths: Sequence[int] | int, unique: bool = False
    ) -> "PackedTensor":
        """Lookup segments in the packed tensor based on provided indices.

        Args:
            packed_idcs (torch.Tensor): A tensor containing indices to lookup in the packed tensor.
            idcs_lengths (Sequence[int] | int): Lengths of the indices for each segment. If a single integer is
                provided, it is assumed that all segments have the same length.
            unique (bool): If True, returns only unique values from the segments. Defaults to False.
        Returns:
            PackedTensor: A new PackedTensor containing the concatenated segments corresponding to the provided indices.
        """
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
        """Convert the packed tensor to a padded tensor.

        Args:
            pad_value (int): The value to use for padding. Defaults to 0.
        Returns:
            torch.Tensor: A padded tensor where each segment is padded to the length of the longest segment
                in the packed tensor.
        """
        return torch.nn.utils.rnn.pad_sequence(self.segmented_tensor, batch_first=True, padding_value=pad_value)
