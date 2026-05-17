import torch
import torch.nn as nn
from gsnn.models.SignedMessagePassing import SignedMessagePassing

class NodeAttention(torch.nn.Module):
    r"""Node-wise channel attention.

    The layer learns a single scalar attention coefficient \(\alpha_{b,n}\) per **node** *n*
    for every sample in the batch *b*.  The coefficient is obtained by first
    aggregating the (optionally weighted) hidden channels that belong to the
    node and then normalising the aggregated scores across all nodes with a
    sigmoid gates per node (no cross-node normalization). The resulting
    attention weights can be:

    1. **Interpreted** - \(\alpha_{b,n}\) tells how important node *n* was for
       the current forward pass.
    2. **Applied** - the coefficients are broadcast back to the individual
       channels that originated from the node and multiplied with the original
       activations, producing an attention-modulated output.

    Parameters
    ----------
    channel_groups : Sequence[int] or Tensor
        A 1-D list/array mapping *global channel index* → *node index*.
        Length equals the total number of hidden channels across all nodes.
    dropout : float, optional (default=0.0)
        Dropout probability applied to the node-level attention weights.
    temperature : float, optional (default=1.0)
        Softmax temperature.  Lower values produce sharper distributions.

    Examples
    --------
    >>> # Suppose we have 2 nodes with 3 channels each (total 6 channels)
    >>> ch_groups = [0, 0, 0, 1, 1, 1]
    >>> attn = NodeAttention(ch_groups, dropout=0.1)
    >>> x = torch.randn(8, 6)  # (batch=8, channels=6)
    >>> out, alpha = attn(x, return_alpha=True)
    >>> out.shape          # same shape as input
    torch.Size([8, 6])
    >>> alpha.shape        # one scalar per node
    torch.Size([8, 2])
    """

    def __init__(self, channel_groups, 
                 dropout: float = 0.0, 
                 temperature: float = 1.0, 
                 channels=16,
                 edge_index=None,
                 edge_weight=None):
        super().__init__()

        # Convert and store channel → node mapping
        self.register_buffer('channel_groups', torch.as_tensor(channel_groups, dtype=torch.long))

        self.n_nodes: int = int(self.channel_groups.max().item() + 1)
        self.dropout: float = float(dropout)
        self.temperature: float = float(temperature)
        
        if (edge_weight is not None) and (edge_index is not None):
            self.signed_message_passing = SignedMessagePassing(edge_weight, edge_index)
        else:
            self.signed_message_passing = torch.nn.Identity()

        # Compute how many channels belong to each node (assume uniform)
        self.channels_per_node: int = int(self.channel_groups.numel() // self.n_nodes)

        # Shared two-layer MLP that maps a vector of node-channels → scalar gate
        self.mlp = nn.Sequential(
            nn.Linear(self.channels_per_node, channels),
            nn.ELU(),
            nn.LayerNorm(channels),
            nn.Linear(channels, 1),
        )

        # will hold the last computed attention for inspection
        self._last_alpha = None

    def forward(self, x: torch.Tensor, *, return_alpha: bool = False):
        """Apply node attention.

        Parameters
        ----------
        x : Tensor of shape (B, C)
            Input activations ordered so that channels belonging to the same
            node are indexed according to `channel_groups`.
        return_alpha : bool, optional (default=False)
            If *True*, the method returns a tuple ``(out, alpha)`` where
            ``alpha`` is the attention matrix of shape *(B, n_nodes)*.

        Returns
        -------
        Tensor or Tuple[Tensor, Tensor]
            The attention-modulated activations (and, optionally, the node
            coefficients).
        """
        x = x.squeeze(-1)

        B, C = x.shape
        if C != self.channel_groups.numel():
            raise ValueError(
                f"Expected input with {self.channel_groups.numel()} channels, got {C}.")

        x_nodes = x.view(B, self.n_nodes, self.channels_per_node)             # (B, N, C_pn)

        alpha = self.mlp(x_nodes).squeeze(-1)                                 # (B, N); logits 

        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)

        # signed message passing (sum of neighbors * edge sign)
        alpha = self.signed_message_passing(alpha)

        alpha_per_channel = alpha.sigmoid().unsqueeze(-1).expand(-1, -1, self.channels_per_node)   # (B,N,C_pn)
        out = (x_nodes * alpha_per_channel).reshape(B, C)                                # (B, C)

        self._last_alpha = alpha

        return (out, alpha) if return_alpha else out