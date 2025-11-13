from typing import Literal

import torch


class Pooler(torch.nn.Module):
    """Applies pooling to the embeddings based on the pooling strategy defined in the configuration."""

    def __init__(self, pooling_strategy: Literal["first", "mean", "max", "sum"] | None) -> None:
        """Initializes the pooler.

        Args:
            pooling_strategy (Literal['first', 'mean', 'max', 'sum'] | None): Pooling strategy to aggregate the
                contextualized embeddings into a single vector.
        """
        super().__init__()
        self.pooling_strategy = pooling_strategy

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Applies optional pooling to the embeddings.

        Args:
            embeddings (torch.Tensor): Query, document, or joint query-document embeddings
            attention_mask (torch.Tensor | None): Query, document, or joint query-document attention mask
        Returns:
            torch.Tensor: (Optionally) pooled embeddings.
        Raises:
            ValueError: If an unknown pooling strategy is passed.
        """
        if self.pooling_strategy is None:
            return embeddings
        if self.pooling_strategy == "first":
            return embeddings[:, [0]]
        if self.pooling_strategy in ("sum", "mean"):
            if attention_mask is not None:
                embeddings = embeddings * attention_mask.unsqueeze(-1)
            embeddings = embeddings.sum(dim=1, keepdim=True)
            if self.pooling_strategy == "mean":
                if attention_mask is not None:
                    embeddings = embeddings / attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
            return embeddings
        if self.pooling_strategy == "max":
            if attention_mask is not None:
                embeddings = embeddings.masked_fill(~attention_mask.bool().unsqueeze(-1), float("-inf"))
            return embeddings.amax(dim=1, keepdim=True)
        raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")


class Sparsifier(torch.nn.Module):
    """Applies sparsification to the embeddings based on the sparsification strategy defined in the configuration."""

    def __init__(self, sparsification_strategy: Literal["relu", "relu_log", "relu_2xlog"] | None) -> None:
        """Initializes the sparsifier.

        Args:
            sparsification_strategy (Literal['relu', 'relu_log', 'relu_2xlog'] | None): Which sparsification strategy
                to apply.
        """
        super().__init__()
        self.sparsification_strategy = sparsification_strategy

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Applies optional sparsification to the embeddings.

        Args:
            embeddings (torch.Tensor): Query, document, or joint query-document embeddings
        Returns:
            torch.Tensor: (Optionally) sparsified embeddings.
        Raises:
            ValueError: If an unknown sparsification strategy is passed.
        """
        if self.sparsification_strategy is None:
            return embeddings
        if self.sparsification_strategy == "relu":
            return torch.relu(embeddings)
        if self.sparsification_strategy == "relu_log":
            return torch.log1p(torch.relu(embeddings))
        if self.sparsification_strategy == "relu_2xlog":
            return torch.log1p(torch.log1p(torch.relu(embeddings)))
        raise ValueError(f"Unknown sparsification strategy: {self.sparsification_strategy}")
