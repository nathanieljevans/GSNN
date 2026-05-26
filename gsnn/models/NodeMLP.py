import torch
import torch.nn as nn


class NodeMLP(nn.Module):
    """Small MLP applied independently to each node's channel vector inside a ResBlock."""

    def __init__(self, in_features: int, hidden_features: int, nonlin, dropout):
        """
        Args:
            in_features: Channels per node (width of each node's slice).
            hidden_features: Hidden width of the two-layer MLP.
            nonlin: Activation module class (e.g. ``torch.nn.ELU``).
            dropout: Dropout probability between layers.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nonlin(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(batch, num_nodes, channels_per_node)``; returns same shape."""
        x = self.mlp(x)
        return x
